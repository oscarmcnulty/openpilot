import os
import threading
import time
import unittest
import numpy as np
import usb1

from openpilot.selfdrive.modeld.remote_model import (RemotePolicyClient, RemotePolicyServer, RemoteModelError, FdLink, UsbLink,
                                                     LinkTimeout, CHUNK, MSG_OK, MSG_HELLO, HEADER, MAGIC)

FRAME_SKIP = 4
INPUT_SHAPES = {'img': (1, 12, 4, 8), 'big_img': (1, 12, 4, 8), 'features_buffer': (1, 24, 3),
                'desire_pulse': (1, 25, 2), 'traffic_convention': (1, 2), 'action_t': (1, 2)}
OUTPUT_SLICES = {'a': slice(0, 2), 'hidden_state': slice(2, 5), 'pad': slice(-2, None)}
METADATA = {'input_shapes': INPUT_SHAPES, 'output_slices': OUTPUT_SLICES, 'model': 'stub'}


class StubPolicy:
  warped_shape = (2, 6, 4, 8)
  packed_len = 2 + 2 + 2 + 3

  def __init__(self):
    self.delay = 0.0
    self.reset()

  def reset(self):
    self.runs = 0

  def run(self, warped, packed):
    self.runs += 1
    time.sleep(self.delay)
    return np.array([warped.sum(), packed.sum(), self.runs, warped[1, 0, 0, 0], 0.], dtype=np.float32)


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

  def test_bad_magic(self):
    a, b = pipe_links()
    a.write(b'XXXX' + bytes(5))
    with self.assertRaises(ConnectionError):
      b.recv()
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
    RemotePolicyServer(StubPolicy(), METADATA, FRAME_SKIP).serve(self.link)


class TestRemoteModel(unittest.TestCase):
  def setUp(self):
    self.policy = StubPolicy()
    self.server = RemotePolicyServer(self.policy, METADATA, FRAME_SKIP)
    self.device_link, self.host_link = pipe_links()
    self.thread = threading.Thread(target=self.server.serve, args=(self.host_link,), daemon=True)
    self.thread.start()
    self.client = RemotePolicyClient(link=self.device_link, run_timeout=2.0, setup_timeout=2.0)

  def tearDown(self):
    self.client.close()
    self.host_link.close()
    self.thread.join(2.0)
    self.assertFalse(self.thread.is_alive())

  def test_hello_metadata(self):
    md = self.client.connect(FRAME_SKIP)
    self.assertEqual(md['input_shapes'], INPUT_SHAPES)
    self.assertEqual(md['output_slices'], OUTPUT_SLICES)

  def test_run_and_reconnect_resets(self):
    self.client.connect(FRAME_SKIP)
    warped = np.full(StubPolicy.warped_shape, 3, dtype=np.uint8)
    warped[1, 0, 0, 0] = 7
    packed = np.arange(StubPolicy.packed_len, dtype=np.float32)
    out = self.client.run(warped, packed)
    self.assertEqual(out.dtype, np.float32)
    self.assertTrue(out.flags.writeable)  # the parser modifies outputs in place
    np.testing.assert_allclose(out, [warped.sum(), packed.sum(), 1, 7, 0])
    self.assertEqual(self.client.run(warped, packed)[2], 2)
    self.client.connect(FRAME_SKIP)
    self.assertEqual(self.client.run(warped, packed)[2], 1)

  def test_bad_inputs_rejected_by_host(self):
    self.client.connect(FRAME_SKIP)
    with self.assertRaises(RemoteModelError):
      self.client.run(np.zeros((2, 6, 4, 4), dtype=np.uint8), np.zeros(StubPolicy.packed_len, dtype=np.float32))

  def test_frame_skip_mismatch(self):
    with self.assertRaises(RemoteModelError):
      self.client.connect(FRAME_SKIP + 1)
    # a rejected hello leaves the link usable
    self.client.connect(FRAME_SKIP)

  def test_timeout_kills_client(self):
    self.client.connect(FRAME_SKIP)
    self.policy.delay = 1.0
    self.client.run_timeout = 0.2
    warped = np.zeros(StubPolicy.warped_shape, dtype=np.uint8)
    packed = np.zeros(StubPolicy.packed_len, dtype=np.float32)
    with self.assertRaises(RemoteModelError):
      self.client.run(warped, packed)
    self.assertTrue(self.client.dead)
    with self.assertRaises(RemoteModelError):
      self.client.connect(FRAME_SKIP)


if __name__ == "__main__":
  unittest.main()
