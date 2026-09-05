"""Run the driving policy on a machine plugged into the aux USB-C port, e.g. a Mac with an Apple GPU.

modeld keeps the warp on the device GPU and ships the warped frames plus the small float inputs
to the host, which owns the temporal queues and runs the policy JIT, exactly like the chestnut USB
GPU path. The device is a USB gadget (see system/hardware/tici/remote_model_gadget.py) with one bulk
IN and one bulk OUT endpoint; the host drives it with libusb (remote_model_server.py).

Messages: 4 byte magic, 1 byte kind, 4 byte little-endian payload length, then the payload in
transfers of at most CHUNK bytes. Both sides always know the exact size of the next transfer, which
bulk USB needs.
  HELLO  device -> host  JSON {version, frame_skip}
  OK     host -> device  JSON model metadata (in reply to HELLO), empty (RESET), or float32 output (RUN)
  RESET  device -> host  clear the temporal queues, done on connect and after warmup
  RUN    device -> host  warped uint8 bytes followed by packed float32 bytes
  ERR    host -> device  UTF-8 error message
"""
import json
import os
import queue
import struct
import threading
import numpy as np

from openpilot.common.hardware.usb import usb_gadget_configured
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.tici.remote_model_gadget import EP_IN_FILE, EP_OUT_FILE

PROTOCOL_VERSION = 1
SETUP_TIMEOUT = 60.0  # hello and reset: the host may still be loading weights
RUN_TIMEOUT = 0.15    # per frame, model runs at 20Hz. a miss falls modeld back to the small model
CHUNK = 64 * 1024

MAGIC = b'OPRM'
HEADER = struct.Struct('<4sBI')
MSG_HELLO, MSG_OK, MSG_RESET, MSG_RUN, MSG_ERR = range(1, 6)


class RemoteModelError(RuntimeError):
  pass


class LinkTimeout(Exception):
  """Nothing arrived on the link within the timeout. Only raised for the header, so no message was lost."""


def slices_to_json(output_slices: dict[str, slice]) -> dict[str, list]:
  return {k: [v.start, v.stop] for k, v in output_slices.items()}


def slices_from_json(d: dict[str, list]) -> dict[str, slice]:
  return {k: slice(start, stop) for k, (start, stop) in d.items()}


def chunk_sizes(n: int) -> list[int]:
  return [min(CHUNK, n - off) for off in range(0, n, CHUNK)]


class Link:
  """Message framing over a byte transport. Subclasses implement one transfer of an exact size each way."""
  def write(self, data: bytes) -> None:
    raise NotImplementedError

  def read(self, n: int, header: bool = False) -> bytes:
    raise NotImplementedError

  def close(self) -> None:
    pass

  def send(self, kind: int, payload: bytes = b'') -> None:
    self.write(HEADER.pack(MAGIC, kind, len(payload)))
    for off, size in zip(range(0, len(payload), CHUNK), chunk_sizes(len(payload)), strict=True):
      self.write(payload[off:off + size])

  def recv(self) -> tuple[int, bytes]:
    magic, kind, length = HEADER.unpack(self.read(HEADER.size, header=True))
    if magic != MAGIC:
      raise ConnectionError(f"bad magic {magic!r}, link out of sync")
    return kind, b''.join(self.read(size) for size in chunk_sizes(length))


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
    for fd in (self.read_fd, self.write_fd):
      try:
        os.close(fd)
      except OSError:
        pass


class UsbLink(Link):
  """Host side: libusb bulk transfers to the gadget. timeout is in seconds."""
  def __init__(self, handle, ep_in: int, ep_out: int, timeout: float = 1.0):
    import usb1
    self.usb1 = usb1
    self.handle, self.ep_in, self.ep_out = handle, ep_in, ep_out
    self.timeout_ms = int(timeout * 1000)

  # every libusb failure becomes a ConnectionError so the serve loop drops the link and the device gets reopened.
  # a write times out when the device stopped reading (modeld died mid-request), NoDevice/IO when it unplugged or rebooted.
  def write(self, data: bytes) -> None:
    view = memoryview(data)
    try:
      while view:
        view = view[self.handle.bulkWrite(self.ep_out, view, timeout=self.timeout_ms):]
    except self.usb1.USBError as e:
      raise ConnectionError(f"usb write failed: {e}") from e

  def read(self, n: int, header: bool = False) -> bytes:
    try:
      data = self.handle.bulkRead(self.ep_in, n, timeout=self.timeout_ms)
    except self.usb1.USBErrorTimeout:
      if header:
        raise LinkTimeout() from None
      raise ConnectionError("timeout mid-message") from None
    except self.usb1.USBError as e:
      raise ConnectionError(f"usb read failed: {e}") from e
    if len(data) != n:
      raise ConnectionError(f"short transfer {len(data)} of {n}")
    return bytes(data)

  def close(self) -> None:
    try:
      self.handle.releaseInterface(0)
      self.handle.close()
    except Exception:
      pass


def remote_model_available() -> bool:
  if os.getenv("NO_REMOTE_MODEL"):  # force the on-device model, e.g. for A/B replays with the host still attached
    return False
  return usb_gadget_configured() and os.path.exists(EP_IN_FILE) and os.path.exists(EP_OUT_FILE)


class RemotePolicyClient:
  """Device side. Endpoint I/O blocks with no timeout when the host stops servicing the link, so it runs
  on a thread and requests wait on a queue with a deadline. A timed-out client is dead: modeld falls
  back to the small model and a fresh client is made on the next start."""
  def __init__(self, link: Link | None = None, run_timeout: float = RUN_TIMEOUT, setup_timeout: float = SETUP_TIMEOUT):
    self.link = link if link is not None else FdLink.open_gadget()
    self.run_timeout, self.setup_timeout = run_timeout, setup_timeout
    self.dead = False
    self.warped_shape: tuple[int, ...] = ()
    self.packed_len = 0
    self.out_len = 0
    self._requests: queue.Queue = queue.Queue()
    self._replies: queue.Queue = queue.Queue()
    self._thread = threading.Thread(target=self._io_loop, daemon=True)
    self._thread.start()

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
    resp_kind, resp = reply
    if resp_kind == MSG_ERR:
      raise RemoteModelError(resp.decode(errors='replace'))
    if resp_kind != MSG_OK:
      raise RemoteModelError(f"unexpected reply {resp_kind}")
    return resp

  def connect(self, frame_skip: int) -> dict:
    """Returns the model metadata dict: input_shapes, output_slices."""
    hello = json.dumps({'version': PROTOCOL_VERSION, 'frame_skip': frame_skip}).encode()
    md = json.loads(self._request(MSG_HELLO, hello, self.setup_timeout))
    if md['frame_skip'] != frame_skip:
      raise RemoteModelError(f"host frame_skip {md['frame_skip']} != {frame_skip}")
    self.warped_shape = tuple(md['warped_shape'])
    self.packed_len = int(md['packed_len'])
    self.out_len = int(md['out_len'])
    cloudlog.warning(f"remote model connected: {md.get('model')}")
    return {
      'input_shapes': {k: tuple(v) for k, v in md['input_shapes'].items()},
      'output_slices': slices_from_json(md['output_slices']),
    }

  def reset(self) -> None:
    self._request(MSG_RESET, b'', self.setup_timeout)

  def run(self, warped: np.ndarray, packed: np.ndarray) -> np.ndarray:
    if warped.shape != self.warped_shape or warped.dtype != np.uint8:
      raise RemoteModelError(f"bad warped input {warped.shape} {warped.dtype}, want {self.warped_shape} uint8")
    if packed.size != self.packed_len:
      raise RemoteModelError(f"bad packed input len {packed.size}, want {self.packed_len}")
    payload = np.ascontiguousarray(warped).tobytes() + np.ascontiguousarray(packed, dtype=np.float32).tobytes()
    # copy: frombuffer is read-only and the parser works in place
    out = np.frombuffer(self._request(MSG_RUN, payload, self.run_timeout), dtype=np.float32).copy()
    if out.size != self.out_len:
      raise RemoteModelError(f"bad output len {out.size}, want {self.out_len}")
    return out

  def close(self) -> None:
    self.dead = True
    self._requests.put(None)
    self.link.close()


class RemotePolicyServer:
  """Host side, serves one device link at a time.

  policy must provide:
    warped_shape, packed_len: expected input sizes
    reset() -> None: clear the temporal queues
    run(warped: np.ndarray[uint8], packed: np.ndarray[float32]) -> np.ndarray[float32]
  metadata: dict with input_shapes, output_slices (slices), out_len and an optional model name
  """
  def __init__(self, policy, metadata: dict, frame_skip: int):
    self.policy = policy
    self.frame_skip = frame_skip
    self.hello_reply = json.dumps({
      'version': PROTOCOL_VERSION,
      'frame_skip': frame_skip,
      'model': metadata.get('model', ''),
      'input_shapes': {k: list(v) for k, v in metadata['input_shapes'].items()},
      'output_slices': slices_to_json(metadata['output_slices']),
      'warped_shape': list(policy.warped_shape),
      'packed_len': int(policy.packed_len),
      'out_len': int(metadata['out_len']),
    }).encode()
    self.warped_nbytes = int(np.prod(policy.warped_shape))
    self.packed_nbytes = int(policy.packed_len) * 4

  def serve(self, link: Link) -> None:
    """Runs until the link breaks. Idle timeouts just keep waiting for the device."""
    self.policy.reset()
    while True:
      try:
        kind, payload = link.recv()
      except LinkTimeout:
        continue
      except Exception as e:
        print(f"link closed: {e}", flush=True)
        return
      try:
        reply = self._dispatch(kind, payload)
      except RemoteModelError as e:
        reply = (MSG_ERR, str(e).encode())
      except Exception as e:
        print(f"request failed: {e!r}", flush=True)
        reply = (MSG_ERR, repr(e).encode())
      try:
        link.send(*reply)
      except Exception as e:
        print(f"link closed: {e}", flush=True)
        return

  def _dispatch(self, kind: int, payload: bytes) -> tuple[int, bytes]:
    if kind == MSG_HELLO:
      hello = json.loads(payload)
      if hello.get('version') != PROTOCOL_VERSION:
        raise RemoteModelError(f"protocol version {hello.get('version')} != {PROTOCOL_VERSION}")
      if hello.get('frame_skip') != self.frame_skip:
        raise RemoteModelError(f"frame_skip {hello.get('frame_skip')} != host {self.frame_skip}")
      self.policy.reset()
      return MSG_OK, self.hello_reply
    if kind == MSG_RESET:
      self.policy.reset()
      return MSG_OK, b''
    if kind == MSG_RUN:
      if len(payload) != self.warped_nbytes + self.packed_nbytes:
        raise RemoteModelError(f"bad RUN payload {len(payload)} bytes, want {self.warped_nbytes + self.packed_nbytes}")
      warped = np.frombuffer(payload, dtype=np.uint8, count=self.warped_nbytes).reshape(self.policy.warped_shape)
      packed = np.frombuffer(payload, dtype=np.float32, offset=self.warped_nbytes)
      out = self.policy.run(warped, packed)
      return MSG_OK, np.ascontiguousarray(out, dtype=np.float32).tobytes()
    raise RemoteModelError(f"unknown message kind {kind}")
