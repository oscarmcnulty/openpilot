#!/usr/bin/env python3
"""Serve the big driving policy from this machine's GPU to a comma device over USB.

On the Mac, from the openpilot repo root with the venv active:

  python -m openpilot.selfdrive.modeld.remote_model_server            # run in the foreground
  python -m openpilot.selfdrive.modeld.remote_model_server install    # launchd starts it when the comma is plugged in
  python -m openpilot.selfdrive.modeld.remote_model_server uninstall

Defaults to tinygrad's METAL backend with FLOAT16 and BEAM=2 kernel search; override any of them in
the environment. The model is compiled from the onnx on every start. tinygrad keeps compiled kernels in
its cache (~/Library/Caches/tinygrad on macOS), so only the first start pays for compilation and the
search, which take several minutes. The warmup prints per-run timings, which doubles as the benchmark:
the whole 20Hz budget is 50ms and the link takes a share of it.

The server exits after --idle-timeout seconds without a device. With the launchd agent that is the
standby state: launchd matches the gadget's USB ids (IOKit) and starts the server again on the next
attach. Logs go to ~/Library/Logs/openpilot-remote-model.log, and
`launchctl print gui/$UID/ai.comma.openpilot.remote-model` shows the state.
"""
import os
os.environ.setdefault('DEV', 'METAL')
os.environ.setdefault('FLOAT16', '1')
os.environ.setdefault('BEAM', '2')  # kernel search, 56ms -> 35ms per frame on an Apple GPU. slow first compile, cached after

import argparse
import plistlib
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import usb1

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.onnx import OnnxRunner

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.compile_modeld import make_run_policy, make_input_queues, get_policy_npy_shapes, POLICY_INPUTS
from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
from openpilot.selfdrive.modeld.remote_model import RemotePolicyServer, UsbLink
from openpilot.system.hardware.tici.remote_model_gadget import USB_VID, USB_PID, EP_IN, EP_OUT

WARMUP_RUNS = 5
REPO = Path(__file__).resolve().parents[3]
LABEL = "ai.comma.openpilot.remote-model"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG = Path.home() / "Library" / "Logs" / "openpilot-remote-model.log"


class TinygradPolicy:
  def __init__(self, onnx_path: str, frame_skip: int):
    self.metadata = make_metadata_dict(onnx_path)
    self.run_policy = TinyJit(make_run_policy(OnnxRunner(onnx_path), self.metadata, frame_skip), prune=True)
    self.input_shapes = self.metadata['input_shapes']
    self.frame_skip = frame_skip
    self.dev = Device.DEFAULT
    img = self.input_shapes['img']  # (1, 12, 128, 256): two frames of 6 channels each
    self.warped_shape = (2, 6, img[2], img[3])
    self.packed_len = sum(get_policy_npy_shapes(self.input_shapes)[1])
    # one persistent input buffer, written in place each frame: METAL memory is unified, so this is a memcpy with
    # no allocation or copy kernel. as_memoryview() silently returns a copy unless force_zero_copy is set, so prove
    # the view is real by reading back rather than serve a model that never sees its input.
    self.warped_t = Tensor.zeros(*self.warped_shape, dtype=dtypes.uint8, device=self.dev).contiguous().realize()
    self.warped_mv = self.warped_t.uop.buffer.as_memoryview(force_zero_copy=True)
    probe = np.random.default_rng(1).integers(0, 256, self.warped_mv.nbytes, dtype=np.uint8)
    self.warped_mv[:] = memoryview(probe).cast('B')
    if not np.array_equal(self.warped_t.numpy().reshape(-1), probe):
      raise RuntimeError(f"input buffer is not zero-copy on {self.dev}")
    self.reset()

  def reset(self) -> None:
    self.input_queues, self.npy = make_input_queues(self.input_shapes, self.frame_skip, device=self.dev)

  def run(self, warped: np.ndarray, packed: np.ndarray) -> np.ndarray:
    self.npy['packed'][:] = packed
    self.warped_mv[:] = memoryview(np.ascontiguousarray(warped)).cast('B')
    out, = self.run_policy(**{k: self.input_queues[k] for k in POLICY_INPUTS}, warped=self.warped_t)
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
        handle.claimInterface(0)
        for ep in (EP_IN, EP_OUT):
          handle.clearHalt(ep)  # reset the data toggles after a modeld that died mid-transfer
      except usb1.USBError as e:
        print(f"device found but could not be opened: {e}", flush=True)
        handle.close()
        return None
      return handle
  return None


def serve(onnx: str, idle_timeout: float) -> None:
  if sys.platform == 'darwin':  # no idle sleep while serving; caffeinate cannot stop a lid-close sleep
    subprocess.Popen(['caffeinate', '-dimsw', str(os.getpid())])

  frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
  st = time.perf_counter()
  print(f"compiling {onnx} on {Device.DEFAULT} ...", flush=True)
  policy = TinygradPolicy(onnx, frame_skip)
  print(f"loaded in {time.perf_counter() - st:.1f}s, warming up", flush=True)
  policy.warmup()
  server = RemotePolicyServer(policy, policy.metadata, frame_skip)

  print(f"waiting for the device ({USB_VID:04x}:{USB_PID:04x}) on USB", flush=True)
  idle_since = time.monotonic()
  with usb1.USBContext() as ctx:
    while True:
      handle = open_device(ctx)
      if handle is None:
        if idle_timeout > 0 and time.monotonic() - idle_since > idle_timeout:
          print(f"no device for {idle_timeout:.0f}s, exiting to standby", flush=True)
          return
        time.sleep(1.0)
        continue
      print("device connected", flush=True)
      link = UsbLink(handle, EP_IN, EP_OUT)
      server.serve(link)
      link.close()
      print("device disconnected", flush=True)
      idle_since = time.monotonic()
      time.sleep(0.5)


def install(action: str) -> None:
  domain = f"gui/{os.getuid()}"
  subprocess.run(["launchctl", "bootout", domain, str(PLIST)], capture_output=True)  # replace an existing agent
  if action == "uninstall":
    PLIST.unlink(missing_ok=True)
    print(f"removed {PLIST}")
    return
  # one match per provider class: modern macOS publishes IOUSBHostDevice, the legacy IOUSBDevice nub still exists
  usb_match = {f"comma-remote-model-{cls}": {"IOProviderClass": cls, "idVendor": USB_VID, "idProduct": USB_PID, "IOMatchLaunchStream": True}
               for cls in ("IOUSBHostDevice", "IOUSBDevice")}
  plist = {
    "Label": LABEL,
    # the venv python by path: an activated shell is not available to launchd
    "ProgramArguments": [str(REPO / ".venv" / "bin" / "python"), "-m", "openpilot.selfdrive.modeld.remote_model_server"],
    "WorkingDirectory": str(REPO),
    "StandardOutPath": str(LOG),
    "StandardErrorPath": str(LOG),
    "ProcessType": "Interactive",  # no App Nap or background throttling for the model
    "LaunchEvents": {"com.apple.iokit.matching": usb_match},
  }
  PLIST.parent.mkdir(parents=True, exist_ok=True)
  LOG.parent.mkdir(parents=True, exist_ok=True)
  PLIST.write_bytes(plistlib.dumps(plist))
  subprocess.run(["launchctl", "bootstrap", domain, str(PLIST)], check=True)
  print(f"installed {PLIST}: starts on USB attach, exits to standby when idle, log: {LOG}")


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('action', nargs='?', default='serve', choices=['serve', 'install', 'uninstall'])
  p.add_argument('--onnx', default=str(Path(__file__).parent / 'models' / 'big_driving_supercombo.onnx'))
  p.add_argument('--idle-timeout', type=float, default=300, help='exit after this many seconds without a device, 0 waits forever')
  args = p.parse_args()
  if args.action == 'serve':
    serve(args.onnx, args.idle_timeout)
  else:
    install(args.action)


if __name__ == "__main__":
  main()
