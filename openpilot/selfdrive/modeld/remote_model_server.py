#!/usr/bin/env python3
"""Serve the big driving policy from this machine's GPU to a comma device over USB.

On the Mac, from the openpilot repo root with the venv active:

  python -m openpilot.selfdrive.modeld.remote_model_server            # run in the foreground
  python -m openpilot.selfdrive.modeld.remote_model_server install    # launchd starts it when the comma is plugged in
  python -m openpilot.selfdrive.modeld.remote_model_server uninstall

Defaults to tinygrad's METAL backend with FLOAT16, BEAM=2 kernel search and JIT=2 (no graph batching, which
modeld's SConscript also uses on METAL); override any of them in the environment. The model is compiled from
the onnx on every start, the same way tinygrad's examples/openpilot/compile_onnx.py builds the device pickles.
The onnx is not in the repo since the eGPU model ships precompiled: fetch the one that matches the device's
big_driving_tinygrad.pkl and pass it with --onnx. tinygrad keeps compiled kernels in
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
os.environ.setdefault('JIT', '2')   # METAL graph batching gives wrong outputs once input contents change, see modeld's SConscript

import argparse
import base64
import pickle
import plistlib
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import usb1

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.dtype import _to_np_dtype
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.onnx import OnnxRunner

from openpilot.selfdrive.modeld.remote_model import RemotePolicyServer, UsbLink
from openpilot.system.hardware.tici.remote_model_gadget import USB_VID, USB_PID, EP_IN, EP_OUT

WARMUP_RUNS = 5
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'tinygrad_repo'))  # tinygrad's openpilot compile helpers, the SConscript puts it on PYTHONPATH too
from examples.openpilot.compile_onnx import onnx_metadata
LABEL = "ai.comma.openpilot.remote-model"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG = Path.home() / "Library" / "Logs" / "openpilot-remote-model.log"


class TinygradPolicy:
  """The model JIT and its recurrent state. Inputs named next_<name> in the outputs are state: they stay on the
  GPU and every run writes them in place. The rest is what the device sends each frame (new_img from its warp,
  plus the small float inputs); those are numpy arrays behind NPY tensors, which the JIT copies to the GPU."""
  def __init__(self, onnx_path: str):
    runner = OnnxRunner(onnx_path)
    properties, output_shapes = onnx_metadata(onnx_path)
    self.output_slices = pickle.loads(base64.b64decode(properties['output_slices']))
    specs = {k: (tuple(t.shape), np.dtype(_to_np_dtype(t.dtype))) for k, t in runner.graph_inputs.items()}
    self.state_pairs = {k: f'next_{k}' for k in specs if f'next_{k}' in output_shapes}
    self.input_shapes = {k: v for k, v in specs.items() if k not in self.state_pairs}
    self.host = {k: np.zeros(shape, dtype=dtype) for k, (shape, dtype) in self.input_shapes.items()}
    self.inputs = {k: Tensor(v, device='NPY') for k, v in self.host.items()}
    self.inputs |= {k: Tensor(np.zeros(*specs[k]), device=Device.DEFAULT).contiguous().realize() for k in self.state_pairs}

    # one eager run for the output shapes, like compile_onnx
    outputs = runner({k: v.to(Device.DEFAULT) for k, v in self.inputs.items()})
    self.outputs = {k: Tensor.empty(*v.shape, dtype=v.dtype, device=Device.DEFAULT).contiguous().realize() for k, v in outputs.items()}
    self.outputs |= {next_k: self.inputs[k] for k, next_k in self.state_pairs.items()}  # state updates in place

    @TinyJit(prune=True)
    def run(output_buffers, **inputs):
      outputs = runner({k: v.to(Device.DEFAULT) for k, v in inputs.items()})
      Tensor.realize(*(output_buffers[k].assign(v) for k, v in outputs.items()))
    self.run_model = run
    self.reset()

  def reset(self) -> None:
    for k in self.state_pairs:
      self.inputs[k].assign(0).realize()

  def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
    for k, v in self.host.items():
      np.copyto(v, inputs[k])
    self.run_model(output_buffers=self.outputs, **self.inputs)
    return self.outputs['outputs'].numpy()[0]

  def warmup(self) -> None:
    """Captures the JIT, reports timings, and refuses to serve a model that ignores its input."""
    rng = np.random.default_rng(0)
    outs = []
    for i in range(WARMUP_RUNS):
      inputs = {k: rng.integers(0, 256, shape, dtype=dtype) if dtype == np.uint8 else rng.standard_normal(shape).astype(dtype)
                for k, (shape, dtype) in self.input_shapes.items()}
      st = time.perf_counter()
      outs.append(self.run(inputs))
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

  st = time.perf_counter()
  print(f"compiling {onnx} on {Device.DEFAULT} ...", flush=True)
  policy = TinygradPolicy(onnx)
  print(f"loaded in {time.perf_counter() - st:.1f}s, warming up", flush=True)
  policy.warmup()
  server = RemotePolicyServer(policy)

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
  p.add_argument('--onnx', default=str(Path(__file__).parent / 'models' / 'big_driving_supercombo.onnx'), help='the big driving model onnx')
  p.add_argument('--idle-timeout', type=float, default=300, help='exit after this many seconds without a device, 0 waits forever')
  args = p.parse_args()
  if args.action == 'serve':
    serve(args.onnx, args.idle_timeout)
  else:
    install(args.action)


if __name__ == "__main__":
  main()
