#!/usr/bin/env python3
"""Serve the big driving policy from this machine's GPU to a comma device over USB.

On the Mac, from the openpilot repo root with the venv active:

  python -m openpilot.selfdrive.modeld.remote_model_server --onnx openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx

Defaults to tinygrad's METAL backend with FLOAT16 and BEAM=2 kernel search; override any of them in
the environment. The model is compiled from the onnx on every start. tinygrad keeps compiled kernels in
its cache (~/Library/Caches/tinygrad on macOS), so only the first start pays for compilation and the
search, which take several minutes.
The warmup prints per-run timings, which doubles as the benchmark: the whole 20Hz budget is 50ms
and the link takes a share of it.

The device presents the USB gadget on its aux USB-C port whenever openpilot is running, so just plug
that port into the Mac. The gadget is dormant while the port is in host mode, e.g. with chestnut attached.
"""
import os
os.environ.setdefault('DEV', 'METAL')
os.environ.setdefault('FLOAT16', '1')
os.environ.setdefault('BEAM', '2')  # kernel search, 56ms -> 35ms per frame on an Apple GPU. slow first compile, cached after

import argparse
import time
import numpy as np
import usb1

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.onnx import OnnxRunner

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.compile_modeld import make_run_policy, make_input_queues, get_policy_npy_shapes, POLICY_INPUTS, read_file_chunked_to_disk
from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
from openpilot.selfdrive.modeld.remote_model import RemotePolicyServer, UsbLink
from openpilot.system.hardware.tici.remote_model_gadget import USB_VID, USB_PID, EP_IN, EP_OUT

WARMUP_RUNS = 5


class TinygradPolicy:
  def __init__(self, run_policy, input_shapes: dict, frame_skip: int):
    self.run_policy = run_policy
    self.input_shapes = input_shapes
    self.frame_skip = frame_skip
    self.dev = Device.DEFAULT
    img = input_shapes['img']  # (1, 12, 128, 256): two frames of 6 channels each
    self.warped_shape = (2, 6, img[2], img[3])
    self.npy_shapes, sizes = get_policy_npy_shapes(input_shapes)
    self.packed_len = sum(sizes)
    self.split_idx = np.cumsum(sizes[:-1])
    self.reset()

  def reset(self) -> None:
    self.input_queues, self.npy = make_input_queues(self.input_shapes, self.frame_skip, device=self.dev)

  def run(self, warped: np.ndarray, packed: np.ndarray) -> np.ndarray:
    for (k, s), chunk in zip(self.npy_shapes.items(), np.split(packed, self.split_idx), strict=True):
      self.npy[k][:] = chunk.reshape(s)
    warped_t = Tensor(warped, device=self.dev).realize()
    out, = self.run_policy(**{k: self.input_queues[k] for k in POLICY_INPUTS}, warped=warped_t)
    return out.numpy()[0]

  def warmup(self, n: int = WARMUP_RUNS) -> int:
    """Captures the JIT and reports timings. Returns the output length."""
    rng = np.random.default_rng(0)
    out = None
    for i in range(n):
      warped = rng.integers(0, 256, self.warped_shape, dtype=np.uint8)
      packed = rng.standard_normal(self.packed_len).astype(np.float32)
      st = time.perf_counter()
      out = self.run(warped, packed)
      print(f"  warmup [{i+1}/{n}] {(time.perf_counter() - st) * 1e3:7.2f} ms", flush=True)
    self.reset()
    assert out is not None
    return out.size


def load_policy(onnx_path: str, frame_skip: int) -> tuple[TinygradPolicy, dict]:
  model_path = onnx_path if os.path.isfile(onnx_path) else read_file_chunked_to_disk(onnx_path)
  metadata = make_metadata_dict(model_path)
  run_policy = TinyJit(make_run_policy(OnnxRunner(model_path), metadata, frame_skip), prune=True)
  return TinygradPolicy(run_policy, metadata['input_shapes'], frame_skip), metadata


def open_device(ctx: usb1.USBContext):
  for dev in ctx.getDeviceIterator(skip_on_error=True):
    if dev.getVendorID() == USB_VID and dev.getProductID() == USB_PID:
      handle = dev.open()
      handle.setAutoDetachKernelDriver(True)
      handle.claimInterface(0)
      for ep in (EP_IN, EP_OUT):
        handle.clearHalt(ep)
      return handle
  return None


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--onnx', default='openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx')
  args = p.parse_args()

  frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
  st = time.perf_counter()
  print(f"compiling {args.onnx} on {Device.DEFAULT} ...", flush=True)
  policy, metadata = load_policy(args.onnx, frame_skip)
  print(f"loaded in {time.perf_counter() - st:.1f}s, warming up", flush=True)
  out_len = policy.warmup()
  print(f"output len {out_len}", flush=True)

  server = RemotePolicyServer(policy, {
    'input_shapes': metadata['input_shapes'],
    'output_slices': metadata['output_slices'],
    'out_len': out_len,
    'model': os.path.basename(args.onnx),
  }, frame_skip)

  print(f"waiting for the device ({USB_VID:04x}:{USB_PID:04x}) on USB", flush=True)
  with usb1.USBContext() as ctx:
    while True:
      try:
        handle = open_device(ctx)
      except usb1.USBError as e:
        print(f"device found but could not be opened: {e}", flush=True)
        handle = None
      if handle is None:
        time.sleep(1.0)
        continue
      print("device connected", flush=True)
      link = UsbLink(handle, EP_IN, EP_OUT)
      try:
        server.serve(link)
      finally:
        link.close()
      print("device disconnected", flush=True)
      time.sleep(0.5)


if __name__ == "__main__":
  main()
