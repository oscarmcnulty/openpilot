#!/usr/bin/env python3
"""Offline pandad timing emulator: the real pandad binary against a fake panda.

pandad runs unmodified (SCHED_FIFO 54 on core 3, like the device) with fake_spi.so preloaded, which
implements the panda SPI protocol with configurable timing and timestamps every HCA_01 the "panda"
receives. A fake card (SCHED_FIFO 53 on core 1) wakes on pandad's `can` publish like the real one,
waits --card-delay-ms, and publishes sendcan, with HCA_01 in every other message. Which pandad
iterations HCA_01 lands in (--parity) is fixed per boot on the device: 0 puts it in the iterations
that run the 10Hz state readout.

The HCA_01 arrival times are scored against the EPS heal rule (vag-ecu-re ecus/eps, hca01_timeout_debounce):
5 ms monitor tick, a frame later than 25 ms restarts healing, 500 ms of on-time frames heals.

  run_emu.py                      # both parities, 20s each
  run_emu.py --parity 0 -d 60     # one run
  run_emu.py --pandad /path/to/modified/pandad
  EMU_IOCTL_SLEEP_US=100 run_emu.py   # fake panda timing knobs, see fake_spi.c
"""
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
PANDAD_DIR = os.path.abspath(os.path.join(HERE, "../.."))
REPO = os.path.abspath(os.path.join(PANDAD_DIR, "../../.."))
FAKE_SO = os.path.join(HERE, "fake_spi.so")


def build_fake_spi():
  src = os.path.join(HERE, "fake_spi.c")
  if not os.path.exists(FAKE_SO) or os.path.getmtime(FAKE_SO) < os.path.getmtime(src):
    subprocess.check_call(["gcc", "-O2", "-Wall", "-Wextra", "-shared", "-fPIC", f"-I{REPO}/panda", src, "-o", FAKE_SO, "-ldl", "-lpthread"])


def read_shm(path):
  with open(path, "rb") as f:
    b = f.read(24)
  if len(b) < 24:
    return 0, 0, 0
  return tuple(int.from_bytes(b[i:i + 8], "little") for i in (0, 8, 16))


# ***** fake card *****

def card(args):
  import openpilot.cereal.messaging as messaging
  os.sched_setaffinity(0, {1})
  os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(53))
  can_sock = messaging.sub_sock("can", timeout=100)
  pm = messaging.PubMaster(["sendcan"])
  rng = random.Random(args.seed)

  pub_log = open(args.shm + ".pub", "w")
  frame = 0
  hca_counter = 0
  hca_offset = None  # card's own HCA parity, fixed once aligned to the requested pandad parity
  while True:
    msgs = messaging.drain_sock_raw(can_sock, wait_for_one=True)
    if not msgs:
      continue
    it, _, it_10hz = read_shm(args.shm)
    if it_10hz == 0:
      continue
    if args.card_mode == "fixed" or hca_offset is None:
      # HCA_01 when (iteration - 10Hz iteration) % 2 == parity. "drain" then just alternates like card does,
      # so a merged or missed can message flips it, "fixed" holds the parity for A/B comparisons.
      hca_offset = (frame + (it - it_10hz) - args.parity) % 2

    delay = args.card_delay_ms + rng.uniform(-args.card_jitter_ms, args.card_jitter_ms)
    if delay > 0:
      time.sleep(delay / 1e3)

    sends = []
    if (frame + hca_offset) % 2 == 0:
      dat = bytearray(8)
      dat[1] = 0x30 | hca_counter
      dat[4] = 3
      hca_counter = (hca_counter + 1) % 16
      sends.append((0x126, bytes(dat), 0))
    if frame % 10 == 0:
      sends.append((0x397, bytes(8), 0))  # LDW_02
    msg = messaging.new_message("sendcan", len(sends))
    for i, (addr, dat, bus) in enumerate(sends):
      msg.sendcan[i].address = addr
      msg.sendcan[i].dat = dat
      msg.sendcan[i].src = bus
    pm.send("sendcan", msg)
    if sends and sends[0][0] == 0x126:
      pub_log.write(f"{time.monotonic_ns()}\n")
      pub_log.flush()
    frame += 1


# ***** EPS model *****

def eps_heal(hca_ns, tick_phase_ms=0.0):
  """hca01_timeout_debounce: returns (time healed after the first frame in s or None, number of heal restarts)"""
  if len(hca_ns) < 2:
    return None, 0
  t0 = hca_ns[0]
  frames = [(t - t0) / 1e6 for t in hca_ns]
  elapsed, heal_acc, restart, restarts = 0.0, 0.0, True, 0
  tick = tick_phase_ms
  i = 1
  while i < len(frames):
    if frames[i] <= tick:
      e = 20.0 if 15.0 <= elapsed <= 25.0 else elapsed
      heal_acc = e if restart else heal_acc + e
      restart = False
      if heal_acc >= 500.0:
        return frames[i] / 1e3, restarts
      elapsed = 0.0
      i += 1
    else:
      elapsed += 5.0
      if elapsed > 25.0 and not restart:
        restart = True
        restarts += 1
      tick += 5.0
  return None, restarts


def analyze(hca_ns):
  gaps = [(b - a) / 1e6 for a, b in zip(hca_ns, hca_ns[1:], strict=False)]
  if not gaps:
    return {"n": len(hca_ns)}
  gaps_sorted = sorted(gaps)
  heal = [eps_heal(hca_ns, ph)[0] for ph in (0.0, 1.25, 2.5, 3.75)]
  return {
    "n": len(hca_ns),
    "gaps_over_25": sum(g > 25 for g in gaps),
    "pct_over_25": 100 * sum(g > 25 for g in gaps) / len(gaps),
    "max_gap": gaps_sorted[-1],
    "p99_gap": gaps_sorted[int(0.99 * (len(gaps) - 1))],
    "mean_gap": sum(gaps) / len(gaps),
    "heal_s": heal,
  }


# ***** one run *****

def run(args, parity, seed):
  prefix = f"emu_{uuid.uuid4().hex[:8]}"
  env = dict(os.environ, OPENPILOT_PREFIX=prefix)
  os.environ["OPENPILOT_PREFIX"] = prefix
  os.makedirs(f"/dev/shm/msgq_{prefix}", exist_ok=True)
  if not os.path.exists("/dev/spidev0.0"):
    open("/dev/spidev0.0", "w").close()  # pandad checks the device exists before opening it

  from openpilot.common.params import Params
  from opendbc.car.car_helpers import interfaces
  from opendbc.car import gen_empty_fingerprint
  params = Params()
  CP = interfaces["AUDI_Q5_MK1"].get_params("AUDI_Q5_MK1", gen_empty_fingerprint(), [], False, False, docs=False)
  params.put("CarParams", CP.to_bytes())
  params.put_bool("FirmwareQueryDone", True)
  params.put_bool("ControlsReady", True)

  work = os.path.join(args.out, prefix)
  os.makedirs(work, exist_ok=True)
  shm, hca_log, trace = (os.path.join(work, n) for n in ("shm", "hca.log", "trace.csv"))
  open(shm, "wb").close()

  # pandad's swaglog, for the pandad_timing lines
  import zmq
  ctx = zmq.Context()
  log_sock = ctx.socket(zmq.PULL)
  log_sock.bind(f"ipc:///tmp/logmessage{prefix}")
  records = []
  stop = threading.Event()

  def collect():
    while not stop.is_set():
      if log_sock.poll(100):
        dat = b"".join(log_sock.recv_multipart())
        records.append((time.monotonic(), dat[1:].decode("utf-8", "replace")))
  collector = threading.Thread(target=collect, daemon=True)
  collector.start()

  # deviceState: onroad
  import openpilot.cereal.messaging as messaging
  pm = messaging.PubMaster(["deviceState"])

  def device_state():
    while not stop.is_set():
      m = messaging.new_message("deviceState")
      m.deviceState.started = True
      pm.send("deviceState", m)
      time.sleep(0.5)
  threading.Thread(target=device_state, daemon=True).start()

  pandad_env = dict(env, LD_PRELOAD=FAKE_SO, BOARDD_SKIP_FW_CHECK="1", EMU_SHM=shm, EMU_HCA_LOG=hca_log)
  if args.trace:
    pandad_env["EMU_TRACE"] = trace
  pandad = subprocess.Popen(["chrt", "-f", "54", "taskset", "-c", "3", args.pandad], cwd=PANDAD_DIR, env=pandad_env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  time.sleep(0.5)
  card_proc = subprocess.Popen([sys.executable, __file__, "--card", "--shm", shm, "--parity", str(parity), "--seed", str(seed),
                                "--card-delay-ms", str(args.card_delay_ms), "--card-jitter-ms", str(args.card_jitter_ms),
                                "--card-mode", args.card_mode], env=env)
  time.sleep(args.duration)

  card_proc.send_signal(signal.SIGTERM)
  pandad.send_signal(signal.SIGINT)
  for p in (card_proc, pandad):
    try:
      p.wait(5)
    except subprocess.TimeoutExpired:
      p.kill()
  time.sleep(0.2)
  stop.set()
  collector.join()
  log_sock.close(linger=0)

  rows = []
  if os.path.exists(hca_log):
    with open(hca_log) as f:
      rows = [line.split() for line in f if line.strip()]
  rows = rows[25:] if len(rows) > 50 else rows  # skip startup
  res = analyze([int(r[0]) for r in rows])
  # the fake card's own HCA_01 publish gaps: the emulator's floor, not pandad's doing
  with open(shm + ".pub") as f:
    pub = [int(x) for x in f if x.strip()][25:]
  res["card_gaps_over_25"] = sum((b - a) > 25e6 for a, b in zip(pub, pub[1:], strict=False))
  # where in the 10Hz cycle HCA_01 arrived: share in the state readout iteration, and changes of iteration parity
  phases = [int(r[2]) for r in rows]
  res["in_readout_iter_pct"] = 100 * sum(p == 0 for p in phases) / max(len(phases), 1) * 5
  res["parity_flips"] = sum((a % 2) != (b % 2) for a, b in zip(phases, phases[1:], strict=False))
  return res, records, work


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--pandad", default=os.path.join(PANDAD_DIR, "pandad"))
  parser.add_argument("-d", "--duration", type=float, default=20.)
  parser.add_argument("--parity", type=int, choices=[0, 1], help="HCA_01 in the 10Hz-readout iterations (0) or the others (1); default both")
  parser.add_argument("--card-delay-ms", type=float, default=2.0, help="card's can -> sendcan processing time")
  parser.add_argument("--card-jitter-ms", type=float, default=0.5)
  parser.add_argument("--card-mode", choices=["fixed", "drain"], default="fixed",
                      help="fixed: HCA_01 parity held vs pandad's iterations; drain: alternate per card step like the real card")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--trace", action="store_true", help="also write a CSV of every SPI transfer")
  parser.add_argument("--timing", action="store_true", help="print pandad's own pandad_timing summary too")
  parser.add_argument("--out", default="/tmp/pandad_emu")
  parser.add_argument("--card", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("--shm", help=argparse.SUPPRESS)
  args = parser.parse_args()

  if args.card:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    card(args)
    return

  build_fake_spi()
  print(f"pandad: {args.pandad}\nfake panda: " + " ".join(f"{k}={v}" for k, v in os.environ.items() if k.startswith("EMU_")))
  print(f"{'parity':>6} {'HCA_01':>7} {'gaps>25ms':>10} {'max gap':>8} {'p99 gap':>8} {'flips':>5} {'in10Hz%':>7} {'card>25':>7}  EPS healed after (s), 4 tick phases")
  for parity in ([args.parity] if args.parity is not None else [0, 1]):
    res, records, work = run(args, parity, args.seed)
    if res.get("n", 0) < 2:
      print(f"{parity:>6}  no HCA_01 received, see {work}")
      continue
    heal = " ".join("never" if h is None else f"{h:.2f}" for h in res["heal_s"])
    print(f"{parity:>6} {res['n']:7d} {res['gaps_over_25']:5d} ({res['pct_over_25']:4.1f}%) {res['max_gap']:7.1f}  {res['p99_gap']:7.1f} "
          f"{res['parity_flips']:5d} {res['in_readout_iter_pct']:7.0f} {res['card_gaps_over_25']:7d}   {heal}")
    if args.timing:
      sys.path.insert(0, PANDAD_DIR)
      from timing_report import to_windows, summarize
      summarize(to_windows(records))
      print()
    with open(os.path.join(work, "result.json"), "w") as f:
      json.dump(res, f)


if __name__ == "__main__":
  main()
