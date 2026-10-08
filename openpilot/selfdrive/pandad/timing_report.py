#!/usr/bin/env python3
"""Summarize the pandad_timing instrumentation (selfdrive/pandad/timing_stats.h) from a route's qlogs.

  timing_report.py "<dongle>/<route>"           # whole route
  timing_report.py "<dongle>/<route>/3:6"       # segments 3-5
  timing_report.py "<dongle>/<route>" --windows # also print every 10s window
"""
import argparse
import json
from collections import defaultdict

from openpilot.tools.lib.logreader import LogReader, ReadMode

REQUESTS = {
  "81_00": "can_recv", "03_00": "can_send",
  "00_d2": "get_state", "00_c2": "get_can_state", "00_f3": "heartbeat", "00_e0": "serial_read",
  "00_b0": "set_ir_pwr", "00_b1": "set_fan_speed", "00_b2": "get_fan_speed", "00_e7": "set_power_saving",
  "00_dc": "set_safety_model", "00_df": "set_alt_experience",
}
LAT_BINS = ["<2", "2-5", "5-10", "10-15", "15-20", ">=20"]


class MeanMax:
  def __init__(self):
    self.n, self.sum, self.max = 0, 0., 0.

  def add(self, n, mean_max):
    mean, mx = mean_max
    self.n += n
    self.sum += n * mean
    self.max = max(self.max, mx)

  def __str__(self):
    return f"{self.sum / self.n if self.n else 0:6.2f} / {self.max:6.2f}"


def to_windows(records):
  """records: (t seconds, swaglog record json) -> one dict per 10s dump: {"spi": ..., "loop": ..., "hca": ..., "t": ...}"""
  windows = []
  cur = {}
  for t, record in records:
    msg = json.loads(record).get("msg", "")
    if not isinstance(msg, str) or not msg.startswith("pandad_timing_"):
      continue
    kind, payload = msg.split(" ", 1)
    kind = kind.removeprefix("pandad_timing_")
    if kind == "spi" and cur:
      windows.append(cur)
      cur = {}
    cur[kind] = json.loads(payload)
    cur["t"] = t
  if cur:
    windows.append(cur)
  return windows


def read(identifier):
  lr = LogReader(identifier, default_mode=ReadMode.QLOG)
  return to_windows((m.logMonoTime / 1e9, m.errorLogMessage) for m in lr if m.which() == "errorLogMessage")


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("route")
  parser.add_argument("--windows", action="store_true", help="print every 10s window")
  args = parser.parse_args()
  summarize(read(args.route), args.windows)


def summarize(windows, show_windows=False):
  if not windows:
    print("no pandad_timing lines found (is the instrumented pandad running?)")
    return
  t0 = windows[0]["t"]

  if show_windows:
    print(f"{'t':>7} {'hca':>4} {'gap>25':>6} {'maxgap':>7}  lat bins {LAT_BINS}  worst (lat lock phase frame%10 busy)")
    for w in windows:
      h = w.get("hca", {})
      worst = h.get("worst", [])
      ws = f"{worst[0]['lat']:5.1f} {worst[0]['lock']:5.1f} {worst[0]['phase']:5.1f} {worst[0]['frame_mod10']} {worst[0]['main_busy']}" if worst else ""
      print(f"{w['t'] - t0:7.1f} {h.get('n', 0):4d} {h.get('gaps_over_25', 0):6d} {h.get('max_gap_ms', 0):7.1f}  {h.get('lat_bins')}  {ws}")
    print()

  # ***** HCA_01 sends *****
  hca_n, gaps, max_gap = 0, 0, 0.
  bins = [0] * 6
  lat, recv, lock, xfer = MeanMax(), MeanMax(), MeanMax(), MeanMax()
  by_frame = [[0, 0] for _ in range(10)]
  worst = []
  for w in windows:
    h = w.get("hca")
    if not h or h["n"] == 0:
      continue
    hca_n += h["n"]
    gaps += h["gaps_over_25"]
    max_gap = max(max_gap, h["max_gap_ms"])
    bins = [a + b for a, b in zip(bins, h["lat_bins"], strict=True)]
    for acc, key in ((lat, "lat_ms"), (recv, "recv_ms"), (lock, "lock_ms"), (xfer, "xfer_ms")):
      acc.add(h["n"], h[key])
    for i, (n, late) in enumerate(h["by_frame_mod10"]):
      by_frame[i][0] += n
      by_frame[i][1] += late
    worst += h["worst"]

  print(f"HCA_01 sends: {hca_n}, pandad send-to-send gaps > 25 ms: {gaps} ({100 * gaps / max(hca_n, 1):.1f}%), max gap {max_gap:.1f} ms")
  print("  publish -> sent latency (ms): " + "  ".join(f"{b}: {n}" for b, n in zip(LAT_BINS, bins, strict=True)))
  print(f"  {'':22} mean / max ms")
  print(f"  {'publish -> sent':22} {lat}")
  print(f"  {'publish -> received':22} {recv}")
  print(f"  {'waiting for hw_lock':22} {lock}")
  print(f"  {'can_send call':22} {xfer}")
  print("  by main loop frame % 10 when received (sends, late >= 5 ms): " +
        "  ".join(f"{i}: {n}/{late}" for i, (n, late) in enumerate(by_frame)))
  print("  worst: " + "; ".join(f"{x['lat']:.1f} ms (lock {x['lock']:.1f}, phase {x['phase']:.1f}, frame%10 {x['frame_mod10']}, busy {x['main_busy']})"
                                for x in sorted(worst, key=lambda x: -x["lat"])[:5]))

  # ***** main loop *****
  print("\nmain loop iterations   n      run ms mean / max     transfers mean / max   lagged")
  for kind in ("100hz", "20hz", "10hz", "2hz"):
    n, lagged = 0, 0
    run, xfers = MeanMax(), MeanMax()
    for w in windows:
      lp = w.get("loop", {}).get(kind)
      if lp:
        n += lp["n"]
        lagged += lp["lagged"]
        run.add(lp["n"], lp["run_ms"])
        xfers.add(lp["n"], lp["xfers"])
    print(f"  {kind:6} {n:12d}   {run}          {xfers}      {lagged}")

  # ***** SPI transfers *****
  phases = ["lock", "pre_ta", "hdr", "hack", "post_ta", "data", "dack", "rx", "total"]
  tags = defaultdict(lambda: {"n": 0, "fail": 0, "ms": defaultdict(MeanMax), "hack_polls": MeanMax(), "dack_polls": MeanMax()})
  for w in windows:
    for key, s in w.get("spi", {}).items():
      t = tags[key]
      t["n"] += s["n"]
      t["fail"] += s["fail"]
      for p in phases:
        t["ms"][p].add(s["n"], s["ms"][p])
      t["hack_polls"].add(s["n"], s["hack_polls"])
      t["dack_polls"].add(s["n"], s["dack_polls"])

  print("\nSPI transfers, mean / max ms per phase (lock = waiting for hw_lock, ta = turnaround busy-wait, polls = ACK lltransfers)")
  print(f"  {'transfer':18} {'n':>8} {'fail':>5}  " + "  ".join(f"{p:>15}" for p in phases) + f"  {'hack polls':>15}  {'dack polls':>15}")
  for key, t in sorted(tags.items(), key=lambda kv: -kv[1]["n"]):
    name = REQUESTS.get(key, key)
    print(f"  {name:18} {t['n']:8d} {t['fail']:5d}  " + "  ".join(f"{t['ms'][p]!s:>15}" for p in phases) +
          f"  {t['hack_polls']!s:>15}  {t['dack_polls']!s:>15}")


if __name__ == "__main__":
  main()
