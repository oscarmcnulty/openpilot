import os
import threading
import time
import unittest
import numpy as np
import usb1

from openpilot.selfdrive.modeld.remote_model import (RemotePolicyClient, RemotePolicyServer, RemoteModelError, FdLink, UsbLink,
                                                     LinkTimeout, CHUNK, MSG_OK, MSG_HELLO, HEADER, MAGIC)

INPUT_SHAPES = {'new_img': ((2, 6, 4, 8), np.dtype(np.uint8)), 'desire': ((1, 8), np.dtype(np.float32)),
                'traffic_convention': ((1, 2), np.dtype(np.float32))}
OUTPUT_SLICES = {'a': slice(0, 2), 'hidden_state': slice(2, 5), 'pad': slice(-2, None)}
METADATA = {'input_shapes': INPUT_SHAPES, 'output_slices': OUTPUT_SLICES}


class StubPolicy:
  input_shapes = INPUT_SHAPES
  output_slices = OUTPUT_SLICES

  def __init__(self):
    self.delay = 0.0
    self.reset()

  def reset(self):
    self.runs = 0

  def run(self, inputs):
    self.runs += 1
    time.sleep(self.delay)
    img = inputs['new_img']
    return np.array([img.sum(), inputs['desire'].sum(), self.runs, img[1, 0, 0, 0], inputs['traffic_convention'][0, 1]], dtype=np.float32)


def make_inputs(img_value=0):
  return {k: np.full(shape, img_value if k == 'new_img' else 0, dtype=dtype) for k, (shape, dtype) in INPUT_SHAPES.items()}


def pipe_links() -> tuple[FdLink, FdLink]:
  """Two cross-connected links, like the two ends of the USB cable."""
  a_r, b_w = os.pipe()
  b_r, a_w = os.pipe()
  return FdLink(a_r, a_w), FdLink(b_r, b_w)


class TestFraming(unittest.TestCase):
  def test_roundtrip_large_payload(self):
    a, b = pipe_links()
    payload = os.urandom(CHUNK * 3 + 7)
    t = threading.Thread(target=a.send, args=(MSG_OK, payload))
    t.start()
    self.assertEqual(b.recv(), (MSG_OK, payload))
    t.join()
    a.close()
    b.close()


class FakeUsbHandle:
  """libusb handle stand-in: bulkRead serves queued replies or raises, bulkWrite raises when told to."""
  def __init__(self):
    self.reads: list = []
    self.write_error = None

  def bulkRead(self, ep, n, timeout=0):
    if not self.reads:
      raise usb1.USBErrorTimeout(-7)
    item = self.reads.pop(0)
    if isinstance(item, Exception):
      raise item
    return item[:n]

  def bulkWrite(self, ep, data, timeout=0):
    if self.write_error is not None:
      raise self.write_error
    return len(data)

  def close(self):
    pass


class TestUsbLinkErrors(unittest.TestCase):
  """Every libusb failure must surface as the link's own error types so the server drops the link instead of dying."""
  def setUp(self):
    self.handle = FakeUsbHandle()
    self.link = UsbLink(self.handle, 0x81, 0x01, timeout=0.01)

  def test_header_timeout_is_idle(self):
    with self.assertRaises(LinkTimeout):
      self.link.recv()

  def test_device_gone_closes_link(self):
    for err in (usb1.USBErrorTimeout(-7), usb1.USBErrorNoDevice(-4)):
      self.handle.write_error = err
      with self.assertRaises(ConnectionError):
        self.link.send(MSG_OK, b'x')
    self.handle.reads = [usb1.USBErrorIO(-1)]
    with self.assertRaises(ConnectionError):
      self.link.recv()

  def test_payload_timeout_is_not_idle(self):
    self.handle.reads = [HEADER.pack(MAGIC, MSG_OK, 5)]
    with self.assertRaises(ConnectionError):
      self.link.recv()

  def test_serve_survives_dead_device(self):
    # device sends HELLO, then vanishes before the reply can be written: serve must return, not raise
    self.handle.reads = [HEADER.pack(MAGIC, MSG_HELLO, 2), b'{}']
    self.handle.write_error = usb1.USBErrorTimeout(-7)
    RemotePolicyServer(StubPolicy()).serve(self.link)


class TestRemoteModel(unittest.TestCase):
  def setUp(self):
    self.policy = StubPolicy()
    self.server = RemotePolicyServer(self.policy)
    self.device_link, self.host_link = pipe_links()
    self.thread = threading.Thread(target=self.server.serve, args=(self.host_link,), daemon=True)
    self.thread.start()
    self.client = RemotePolicyClient(link=self.device_link, run_timeout=2.0, setup_timeout=2.0)

  def tearDown(self):
    self.device_link.close()
    self.host_link.close()
    self.thread.join(2.0)
    self.assertFalse(self.thread.is_alive())

  def test_run_and_reconnect_resets(self):
    md = self.client.connect()
    self.assertEqual(md, METADATA)
    inputs = make_inputs(3)
    inputs['new_img'][1, 0, 0, 0] = 7
    inputs['desire'][:] = np.arange(8, dtype=np.float32)
    inputs['traffic_convention'][0, 1] = 1.
    inputs['tfm'] = np.eye(3, dtype=np.float32)  # inputs the host does not take are not sent
    out = self.client.run(inputs)
    self.assertEqual(out.dtype, np.float32)
    self.assertTrue(out.flags.writeable)  # the parser modifies outputs in place
    np.testing.assert_allclose(out, [inputs['new_img'].sum(), 28, 1, 7, 1])
    self.assertEqual(self.client.run(inputs)[2], 2)
    self.client.connect()
    self.assertEqual(self.client.run(inputs)[2], 1)

  def test_version_mismatch(self):
    with self.assertRaises(RemoteModelError):
      self.server._dispatch(MSG_HELLO, b'{"version": 2}')

  def test_bad_payload_size(self):
    self.client.connect()
    inputs = make_inputs()
    inputs['desire'] = np.zeros((1, 9), dtype=np.float32)
    with self.assertRaises(RemoteModelError):
      self.client.run(inputs)
    self.assertFalse(self.client.dead)  # the host rejected the frame, the link is fine
    self.client.run(make_inputs())

  def test_timeout_kills_client(self):
    self.client.connect()
    self.policy.delay = 1.0
    self.client.run_timeout = 0.2
    with self.assertRaises(RemoteModelError):
      self.client.run(make_inputs())
    self.assertTrue(self.client.dead)
    with self.assertRaises(RemoteModelError):
      self.client.connect()


if __name__ == "__main__":
  unittest.main()
