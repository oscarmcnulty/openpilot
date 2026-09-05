#!/usr/bin/env python3
"""Serve the big driving policy from this machine's GPU to a comma device over USB.

On the Mac, from the openpilot repo root with the venv active:

  python -m openpilot.selfdrive.modeld.remote_model_server

Defaults to tinygrad's METAL backend with FLOAT16 and BEAM=2 kernel search; override any of them in
the environment. The model is compiled from the onnx on every start. tinygrad keeps compiled kernels in
its cache (~/Library/Caches/tinygrad on macOS), so only the first start pays for compilation and the
search, which take several minutes. The warmup prints per-run timings, which doubles as the benchmark:
the whole 20Hz budget is 50ms and the link takes a share of it.

The device presents the USB gadget on its aux USB-C port whenever openpilot is running, so just plug
that port into the Mac. The gadget is dormant while the port is in host mode, e.g. with chestnut attached.

With --idle-timeout the server exits after that long without a device, which is how the launchd agent
installed by remote_model_mac_service returns to standby: launchd starts it again on the next USB attach.
"""
import os
os.environ.setdefault('DEV', 'METAL')
os.environ.setdefault('FLOAT16', '1')
os.environ.setdefault('BEAM', '2')  # kernel search, 56ms -> 35ms per frame on an Apple GPU. slow first compile, cached after

import argparse
import subprocess
import sys
import time
import traceback
import numpy as np
import usb1

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.onnx import OnnxRunner

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.compile_modeld import make_run_policy, make_input_queues, get_policy_npy_shapes, POLICY_INPUTS, read_file_chunked_to_disk
from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
from openpilot.selfdrive.modeld.remote_model import RemotePolicyServer, UsbLink
from openpilot.system.hardware.tici.remote_model_gadget import USB_VID, USB_PID, EP_IN, EP_OUT

WARMUP_RUNS = 5


class TinygradPolicy:
  def __init__(self, onnx_path: str, frame_skip: int):
    model_path = onnx_path if os.path.isfile(onnx_path) else read_file_chunked_to_disk(onnx_path)
    self.metadata = make_metadata_dict(model_path) | {'model': os.path.basename(onnx_path)}
    self.run_policy = TinyJit(make_run_policy(OnnxRunner(model_path), self.metadata, frame_skip), prune=True)
    self.input_shapes = self.metadata['input_shapes']
    self.frame_skip = frame_skip
    self.dev = Device.DEFAULT
    img = self.input_shapes['img']  # (1, 12, 128, 256): two frames of 6 channels each
    self.warped_shape = (2, 6, img[2], img[3])
    self.packed_len = sum(get_policy_npy_shapes(self.input_shapes)[1])
    # one persistent input buffer, written in place each frame. METAL memory is unified, so this is a memcpy
    # with no allocation or copy kernel. as_memoryview() silently returns a copy unless force_zero_copy is
    # set, so prove the view is real by reading back; devices without a host-visible buffer copy per frame.
    self.warped_t = Tensor.zeros(*self.warped_shape, dtype=dtypes.uint8, device=self.dev).contiguous().realize()
    self.warped_mv: memoryview | None = None
    try:
      mv = self.warped_t.uop.buffer.as_memoryview(force_zero_copy=True)
      probe = np.random.default_rng(1).integers(0, 256, mv.nbytes, dtype=np.uint8)
      mv[:] = memoryview(probe).cast('B')
      if np.array_equal(self.warped_t.numpy().reshape(-1), probe):
        self.warped_mv = mv
      else:
        print(f"input buffer readback mismatch on {self.dev}, copying per frame", flush=True)
    except Exception as e:
      print(f"no zero-copy input buffer on {self.dev} ({e!r}), copying per frame", flush=True)
    self.reset()

  def reset(self) -> None:
    self.input_queues, self.npy = make_input_queues(self.input_shapes, self.frame_skip, device=self.dev)

  def run(self, warped: np.ndarray, packed: np.ndarray) -> np.ndarray:
    self.npy['packed'][:] = packed
    if self.warped_mv is not None:
      self.warped_mv[:] = memoryview(np.ascontiguousarray(warped)).cast('B')
      warped_t = self.warped_t
    else:
      warped_t = Tensor(warped, device=self.dev).realize()
    out, = self.run_policy(**{k: self.input_queues[k] for k in POLICY_INPUTS}, warped=warped_t)
    return out.numpy()[0]

  def warmup(self) -> None:
    """Captures the JIT, reports timings, and refuses to serve a model that ignores its input."""
    rng = np.random.default_rng(0)
    outs = []
    for i in range(WARMUP_RUNS):
      warped = rng.integers(0, 256, self.warped_shape, dtype=np.uint8)
      packed = rng.standard_normal(self.packed_len).astype(np.float32)
      st = time.perf_counter()
      outs.append(self.run(warped, packed))
      print(f"  warmup [{i+1}/{WARMUP_RUNS}] {(time.perf_counter() - st) * 1e3:7.2f} ms", flush=True)
    # every warmup input was different, so identical outputs mean the model never saw them
    if any(np.array_equal(outs[-1], o) for o in outs[:-1]):
      raise RuntimeError("model output does not depend on the input, refusing to serve")
    if not all(np.all(np.isfinite(o)) for o in outs):
      raise RuntimeError("model output is not finite, refusing to serve")
    self.reset()


def open_device(ctx: usb1.USBContext):
  for dev in ctx.getDeviceIterator(skip_on_error=True):
    if dev.getVendorID() == USB_VID and dev.getProductID() == USB_PID:
      handle = dev.open()
      try:
        handle.setAutoDetachKernelDriver(True)
        handle.claimInterface(0)
        for ep in (EP_IN, EP_OUT):
          handle.clearHalt(ep)
      except usb1.USBError:
        handle.close()
        raise
      return handle
  return None


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--onnx', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'big_driving_supercombo.onnx'))
  p.add_argument('--idle-timeout', type=float, default=0, help='exit after this many seconds without a device, 0 waits forever')
  args = p.parse_args()

  if sys.platform == 'darwin':  # no idle sleep while serving; caffeinate cannot stop a lid-close sleep
    subprocess.Popen(['caffeinate', '-dimsw', str(os.getpid())])

  frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
  st = time.perf_counter()
  print(f"compiling {args.onnx} on {Device.DEFAULT} ...", flush=True)
  policy = TinygradPolicy(args.onnx, frame_skip)
  print(f"loaded in {time.perf_counter() - st:.1f}s, warming up", flush=True)
  policy.warmup()
  server = RemotePolicyServer(policy, policy.metadata, frame_skip)

  print(f"waiting for the device ({USB_VID:04x}:{USB_PID:04x}) on USB", flush=True)
  idle_since = time.monotonic()
  with usb1.USBContext() as ctx:
    while True:
      try:
        handle = open_device(ctx)
      except usb1.USBError as e:
        print(f"device found but could not be opened: {e}", flush=True)
        handle = None
      if handle is None:
        if args.idle_timeout > 0 and time.monotonic() - idle_since > args.idle_timeout:
          print(f"no device for {args.idle_timeout:.0f}s, exiting to standby", flush=True)
          return
        time.sleep(1.0)
        continue
      print("device connected", flush=True)
      link = UsbLink(handle, EP_IN, EP_OUT)
      try:
        server.serve(link)
      except Exception:
        # a policy or runtime failure must not take the server down, the device falls back to its own model
        # for this drive and reconnects on the next start
        traceback.print_exc()
      finally:
        link.close()
      print("device disconnected", flush=True)
      idle_since = time.monotonic()
      time.sleep(0.5)


if __name__ == "__main__":
  main()
