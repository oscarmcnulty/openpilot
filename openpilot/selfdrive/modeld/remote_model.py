"""Run the driving policy on a machine plugged into the aux USB-C port, e.g. a Mac with an Apple GPU.

modeld keeps the warp on the device GPU and ships the warped frames (new_img) plus the small float
inputs to the host, which owns the model's recurrent state and runs the model JIT, like the chestnut
USB GPU path. The device is a USB gadget (see system/hardware/tici/remote_model_gadget.py) with one bulk
IN and one bulk OUT endpoint; the host drives it with libusb (remote_model_server.py).

Messages: 4 byte magic, 1 byte kind, 4 byte little-endian payload length, then the payload in
transfers of at most CHUNK bytes. Both sides always know the exact size of the next transfer, which
bulk USB needs.
  HELLO  device -> host  JSON {version}; also resets the recurrent state
  OK     host -> device  JSON {input_shapes, output_slices} (in reply to HELLO) or float32 output (RUN).
                         input_shapes are the inputs the device sends, {name: [shape, dtype]} in payload order
  RUN    device -> host  each input's bytes, in input_shapes order
  ERR    host -> device  UTF-8 error message
"""
import json
import os
import queue
import struct
import threading
import numpy as np
import usb1

from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.tici.remote_model_gadget import EP_IN_FILE, EP_OUT_FILE

PROTOCOL_VERSION = 3
SETUP_TIMEOUT = 60.0  # hello: the host may still be loading weights
RUN_TIMEOUT = 0.15    # per frame, model runs at 20Hz. a miss falls modeld back to the small model
CHUNK = 64 * 1024

MAGIC = b'OPRM'
HEADER = struct.Struct('<4sBI')
MSG_HELLO, MSG_OK, MSG_RUN, MSG_ERR = range(1, 5)


class RemoteModelError(RuntimeError):
  pass


class LinkTimeout(Exception):
  """Nothing arrived on the link within the timeout. Only raised for the header, so no message was lost."""


def input_nbytes(input_shapes: dict[str, tuple[tuple[int, ...], np.dtype]]) -> int:
  return sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in input_shapes.values())


def slices_to_json(output_slices: dict[str, slice]) -> dict[str, list]:
  return {k: [v.start, v.stop] for k, v in output_slices.items()}


def slices_from_json(d: dict[str, list]) -> dict[str, slice]:
  return {k: slice(start, stop) for k, (start, stop) in d.items()}


class Link:
  """Message framing over a byte transport. Subclasses implement one transfer of an exact size each way."""
  def write(self, data: bytes) -> None:
    raise NotImplementedError

  def read(self, n: int, header: bool = False) -> bytes:
    raise NotImplementedError

  def send(self, kind: int, payload: bytes = b'') -> None:
    self.write(HEADER.pack(MAGIC, kind, len(payload)))
    for off in range(0, len(payload), CHUNK):
      self.write(payload[off:off + CHUNK])

  def recv(self) -> tuple[int, bytes]:
    magic, kind, length = HEADER.unpack(self.read(HEADER.size, header=True))
    if magic != MAGIC:
      raise ConnectionError(f"bad magic {magic!r}, link out of sync")
    return kind, b''.join(self.read(min(CHUNK, length - off)) for off in range(0, length, CHUNK))


class FdLink(Link):
  """Device side: FunctionFS endpoint files. Also works on pipes, which the tests use."""
  def __init__(self, read_fd: int, write_fd: int):
    self.read_fd, self.write_fd = read_fd, write_fd

  @classmethod
  def open_gadget(cls) -> "FdLink":
    # ep1 is IN (device writes), ep2 is OUT (device reads)
    return cls(os.open(EP_OUT_FILE, os.O_RDONLY), os.open(EP_IN_FILE, os.O_WRONLY))

  def write(self, data: bytes) -> None:
    view = memoryview(data)
    while view:
      view = view[os.write(self.write_fd, view):]

  def read(self, n: int, header: bool = False) -> bytes:
    buf = bytearray()
    while len(buf) < n:
      part = os.read(self.read_fd, n - len(buf))
      if not part:
        raise ConnectionError("link closed")
      buf += part
    return bytes(buf)

  def close(self) -> None:
    os.close(self.read_fd)
    os.close(self.write_fd)


class UsbLink(Link):
  """Host side: libusb bulk transfers to the gadget. Every libusb failure becomes a ConnectionError so the
  serve loop drops the link and reopens the device: a write times out when the device stopped reading
  (modeld died mid-request), NoDevice/IO when it unplugged or rebooted."""
  def __init__(self, handle, ep_in: int, ep_out: int, timeout: float = 1.0):
    self.handle, self.ep_in, self.ep_out = handle, ep_in, ep_out
    self.timeout_ms = int(timeout * 1000)

  def write(self, data: bytes) -> None:
    view = memoryview(data)
    try:
      while view:
        view = view[self.handle.bulkWrite(self.ep_out, view, timeout=self.timeout_ms):]
    except usb1.USBError as e:
      raise ConnectionError(f"usb write failed: {e}") from e

  def read(self, n: int, header: bool = False) -> bytes:
    try:
      data = self.handle.bulkRead(self.ep_in, n, timeout=self.timeout_ms)
    except usb1.USBErrorTimeout:
      if header:
        raise LinkTimeout() from None
      raise ConnectionError("timeout mid-message") from None
    except usb1.USBError as e:
      raise ConnectionError(f"usb read failed: {e}") from e
    if len(data) != n:
      raise ConnectionError(f"short transfer {len(data)} of {n}")
    return bytes(data)

  def close(self) -> None:
    self.handle.close()


class RemotePolicyClient:
  """Device side. Endpoint I/O blocks with no timeout when the host stops servicing the link, so it runs
  on a thread and requests wait on a queue with a deadline. A timed-out client is dead: modeld falls
  back to the small model and a fresh client is made on the next start."""
  def __init__(self, link: Link | None = None, run_timeout: float = RUN_TIMEOUT, setup_timeout: float = SETUP_TIMEOUT):
    self.link = link if link is not None else FdLink.open_gadget()
    self.run_timeout, self.setup_timeout = run_timeout, setup_timeout
    self.dead = False
    self.input_shapes: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    self._requests: queue.Queue = queue.Queue()
    self._replies: queue.Queue = queue.Queue()
    threading.Thread(target=self._io_loop, daemon=True).start()

  def _io_loop(self) -> None:
    while (req := self._requests.get()) is not None:
      try:
        self.link.send(*req)
        self._replies.put(self.link.recv())
      except Exception as e:
        self._replies.put(e)
        return

  def _request(self, kind: int, payload: bytes, timeout: float) -> bytes:
    if self.dead:
      raise RemoteModelError("remote model link is dead")
    self._requests.put((kind, payload))
    try:
      reply = self._replies.get(timeout=timeout)
    except queue.Empty:
      self.dead = True
      raise RemoteModelError(f"remote model timed out after {timeout}s") from None
    if isinstance(reply, Exception):
      self.dead = True
      raise RemoteModelError(f"remote model link failed: {reply!r}") from reply
    kind, payload = reply
    if kind != MSG_OK:
      raise RemoteModelError(payload.decode(errors='replace') if kind == MSG_ERR else f"unexpected reply {kind}")
    return payload

  def connect(self) -> dict:
    """Resets the host's recurrent state. Returns the model metadata: input_shapes, output_slices."""
    hello = json.dumps({'version': PROTOCOL_VERSION}).encode()
    md = json.loads(self._request(MSG_HELLO, hello, self.setup_timeout))
    cloudlog.warning("remote model connected")
    self.input_shapes = {k: (tuple(shape), np.dtype(dtype)) for k, (shape, dtype) in md['input_shapes'].items()}
    return {'input_shapes': self.input_shapes, 'output_slices': slices_from_json(md['output_slices'])}

  def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
    payload = b''.join(np.ascontiguousarray(inputs[k], dtype=dtype).tobytes() for k, (_, dtype) in self.input_shapes.items())
    # copy: frombuffer is read-only and the parser works in place
    return np.frombuffer(self._request(MSG_RUN, payload, self.run_timeout), dtype=np.float32).copy()


class RemotePolicyServer:
  """Host side, serves one device link at a time.

  policy must provide:
    input_shapes: {name: (shape, np.dtype)}, the inputs the device sends, in payload order
    output_slices: {name: slice} into the output
    reset() -> None: clear the recurrent state
    run(inputs: dict[str, np.ndarray]) -> np.ndarray[float32]
  """
  def __init__(self, policy):
    self.policy = policy
    self.hello_reply = json.dumps({'input_shapes': {k: [list(shape), dtype.name] for k, (shape, dtype) in policy.input_shapes.items()},
                                   'output_slices': slices_to_json(policy.output_slices)}).encode()
    self.payload_nbytes = input_nbytes(policy.input_shapes)

  def serve(self, link: Link) -> None:
    """Runs until the link breaks. Idle timeouts just keep waiting for the device."""
    while True:
      try:
        try:
          kind, payload = link.recv()
        except LinkTimeout:
          continue
        try:
          reply = self._dispatch(kind, payload)
        except Exception as e:
          reply = (MSG_ERR, str(e).encode())
        link.send(*reply)
      except Exception as e:
        print(f"link closed: {e}", flush=True)
        return

  def _dispatch(self, kind: int, payload: bytes) -> tuple[int, bytes]:
    if kind == MSG_HELLO:
      hello = json.loads(payload)
      if hello.get('version') != PROTOCOL_VERSION:
        raise RemoteModelError(f"protocol version {hello.get('version')} != {PROTOCOL_VERSION}")
      self.policy.reset()
      return MSG_OK, self.hello_reply
    if kind == MSG_RUN:
      if len(payload) != self.payload_nbytes:
        raise RemoteModelError(f"bad RUN payload {len(payload)} bytes, want {self.payload_nbytes}")
      inputs, offset = {}, 0
      for k, (shape, dtype) in self.policy.input_shapes.items():
        inputs[k] = np.frombuffer(payload, dtype=dtype, count=int(np.prod(shape)), offset=offset).reshape(shape)
        offset += inputs[k].nbytes
      return MSG_OK, np.ascontiguousarray(self.policy.run(inputs), dtype=np.float32).tobytes()
    raise RemoteModelError(f"unknown message kind {kind}")
