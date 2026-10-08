#pragma once

// Temporary pandad timing instrumentation for the VW MLB HCA_01 jitter investigation.
// Everything is accumulated in memory and summarized every 10s as three LOGE lines
// ("pandad_timing_spi/loop/hca {json}"), so it lands in the qlog as errorLogMessage.
// Read it back with selfdrive/pandad/timing_report.py.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

#include "common/swaglog.h"
#include "common/timing.h"

namespace pandad_timing {

struct Acc {
  uint64_t n = 0, sum = 0, max = 0;
  void add(uint64_t v) { n++; sum += v; max = std::max(max, v); }
  // "[mean,max]" in ms for ns values, raw for counts
  std::string ms() const {
    char buf[48];
    snprintf(buf, sizeof(buf), "[%.3f,%.3f]", n ? sum / (double)n / 1e6 : 0., max / 1e6);
    return buf;
  }
  std::string cnt() const {
    char buf[48];
    snprintf(buf, sizeof(buf), "[%.2f,%llu]", n ? sum / (double)n : 0., (unsigned long long)max);
    return buf;
  }
};

// ***** SPI transfers, per endpoint/control request *****

enum SpiPhase { LOCK, PRE_TURNAROUND, HEADER, HEADER_ACK, POST_TURNAROUND, DATA, DATA_ACK, RX, TOTAL, N_PHASES };
inline const char *SPI_PHASE_NAMES[N_PHASES] = {"lock", "pre_ta", "hdr", "hack", "post_ta", "data", "dack", "rx", "total"};

struct SpiTag {
  bool used = false;
  uint8_t endpoint = 0, request = 0;
  Acc phase[N_PHASES];
  Acc hack_polls, dack_polls;
  uint64_t fails = 0;
};

// ***** main loop iterations, by the slowest work they ran *****

enum LoopKind { LOOP_100HZ, LOOP_20HZ, LOOP_10HZ, LOOP_2HZ, N_LOOP };
inline const char *LOOP_NAMES[N_LOOP] = {"100hz", "20hz", "10hz", "2hz"};

struct LoopStats {
  Acc run_ns[N_LOOP];  // iteration start -> keepTime, nothing in between sleeps on purpose
  Acc xfers[N_LOOP];   // SPI transfers made by the main thread in the iteration
  uint64_t lagged[N_LOOP] = {};
};

// ***** sendcan messages that carry HCA_01 *****

struct HcaSend {
  float lat_ms = 0, recv_ms = 0, lock_ms = 0, xfer_ms = 0, phase_ms = 0;
  uint32_t frame_mod10 = 0;
  bool main_busy = false;
};

struct HcaStats {
  uint64_t n = 0;
  uint64_t lat_bins[6] = {};  // publish -> sent: <2, 2-5, 5-10, 10-15, 15-20, >=20 ms
  Acc lat_ns, recv_ns, lock_ns, xfer_ns;
  uint64_t gaps_over_25 = 0;  // consecutive sends more than 25ms apart, as pandad sent them
  uint64_t max_gap_ns = 0;
  uint64_t last_done_ns = 0;
  HcaSend worst[5];
  uint32_t by_frame_mod10_n[10] = {}, by_frame_mod10_late[10] = {};  // late = publish -> sent >= 5ms
};

inline std::mutex mtx;  // guards spi and hca, both threads write them
inline SpiTag spi[16];
inline HcaStats hca;
inline LoopStats loop;  // main thread only

// main loop context, read by the send thread
inline std::atomic<uint64_t> iter_start_ns = 0;
inline std::atomic<uint32_t> iter_frame = 0;
inline std::atomic<bool> main_busy = false;

// per thread
inline thread_local uint32_t xfer_count = 0;
inline thread_local uint64_t lock_wait_ns = 0;  // summed over this thread's transfers since reset

inline void record_spi(uint8_t endpoint, uint8_t request, const uint64_t ts[N_PHASES], uint32_t hack_polls, uint32_t dack_polls, bool ok) {
  xfer_count++;
  lock_wait_ns += ts[1] - ts[0];
  std::lock_guard lk(mtx);
  SpiTag *t = nullptr;
  for (auto &s : spi) {
    if (s.used && s.endpoint == endpoint && s.request == request) { t = &s; break; }
    if (!s.used) { s.used = true; s.endpoint = endpoint; s.request = request; t = &s; break; }
  }
  if (t == nullptr) return;
  if (!ok) { t->fails++; return; }
  for (int p = 0; p < TOTAL; p++) t->phase[p].add(ts[p + 1] - ts[p]);
  t->phase[TOTAL].add(ts[N_PHASES - 1] - ts[0]);
  t->hack_polls.add(hack_polls);
  t->dack_polls.add(dack_polls);
}

inline void record_loop(uint32_t frame, uint64_t run_ns, uint32_t xfers, bool lagged) {
  LoopKind k = (frame % 50 == 0) ? LOOP_2HZ : (frame % 10 == 0) ? LOOP_10HZ : (frame % 5 == 0) ? LOOP_20HZ : LOOP_100HZ;
  loop.run_ns[k].add(run_ns);
  loop.xfers[k].add(xfers);
  loop.lagged[k] += lagged;
}

inline void record_hca(uint64_t pub_ns, uint64_t recv_ns, uint64_t start_ns, uint64_t done_ns, uint64_t lock_ns,
                       uint64_t main_start_ns, uint32_t main_frame, bool busy) {
  std::lock_guard lk(mtx);
  HcaSend s;
  s.lat_ms = (done_ns - pub_ns) / 1e6;
  s.recv_ms = (recv_ns - pub_ns) / 1e6;
  s.lock_ms = lock_ns / 1e6;
  s.xfer_ms = (done_ns - start_ns) / 1e6;
  s.phase_ms = (recv_ns - main_start_ns) / 1e6;
  s.frame_mod10 = main_frame % 10;
  s.main_busy = busy;

  hca.n++;
  const float edges[5] = {2, 5, 10, 15, 20};
  int bin = 0;
  while (bin < 5 && s.lat_ms >= edges[bin]) bin++;
  hca.lat_bins[bin]++;
  hca.lat_ns.add(done_ns - pub_ns);
  hca.recv_ns.add(recv_ns - pub_ns);
  hca.lock_ns.add(lock_ns);
  hca.xfer_ns.add(done_ns - start_ns);
  hca.by_frame_mod10_n[s.frame_mod10]++;
  hca.by_frame_mod10_late[s.frame_mod10] += s.lat_ms >= 5;

  if (hca.last_done_ns != 0) {
    uint64_t gap = done_ns - hca.last_done_ns;
    if (gap < 200000000ULL) {  // not across a pause in sending
      hca.gaps_over_25 += gap > 25000000ULL;
      hca.max_gap_ns = std::max(hca.max_gap_ns, gap);
    }
  }
  hca.last_done_ns = done_ns;

  for (int i = 0; i < 5; i++) {
    if (s.lat_ms > hca.worst[i].lat_ms) {
      for (int j = 4; j > i; j--) hca.worst[j] = hca.worst[j - 1];
      hca.worst[i] = s;
      break;
    }
  }
}

inline void dump() {
  SpiTag spi_copy[16];
  HcaStats hca_copy;
  {
    std::lock_guard lk(mtx);
    std::copy(std::begin(spi), std::end(spi), std::begin(spi_copy));
    hca_copy = hca;
    for (auto &s : spi) s = SpiTag();
    uint64_t last_done = hca.last_done_ns;
    hca = HcaStats();
    hca.last_done_ns = last_done;
  }
  LoopStats loop_copy = loop;
  loop = LoopStats();

  // SPI: "<endpoint>_<request>": {n, fail, ms: {phase: [mean,max]}, polls: {hack, dack}}
  std::string out = "{";
  for (auto &s : spi_copy) {
    if (!s.used) continue;
    char key[16];
    snprintf(key, sizeof(key), "%02x_%02x", s.endpoint, s.request);
    out += std::string(out.size() > 1 ? "," : "") + "\"" + key + "\":{\"n\":" + std::to_string(s.phase[TOTAL].n) +
           ",\"fail\":" + std::to_string(s.fails) + ",\"ms\":{";
    for (int p = 0; p < N_PHASES; p++) {
      out += std::string(p ? "," : "") + "\"" + SPI_PHASE_NAMES[p] + "\":" + s.phase[p].ms();
    }
    out += "},\"hack_polls\":" + s.hack_polls.cnt() + ",\"dack_polls\":" + s.dack_polls.cnt() + "}";
  }
  out += "}";
  LOGE("pandad_timing_spi %s", out.c_str());

  out = "{";
  for (int k = 0; k < N_LOOP; k++) {
    out += std::string(k ? "," : "") + "\"" + LOOP_NAMES[k] + "\":{\"n\":" + std::to_string(loop_copy.run_ns[k].n) +
           ",\"run_ms\":" + loop_copy.run_ns[k].ms() + ",\"xfers\":" + loop_copy.xfers[k].cnt() +
           ",\"lagged\":" + std::to_string(loop_copy.lagged[k]) + "}";
  }
  out += "}";
  LOGE("pandad_timing_loop %s", out.c_str());

  const HcaStats &h = hca_copy;
  out = "{\"n\":" + std::to_string(h.n) + ",\"lat_bins\":[";
  for (int i = 0; i < 6; i++) out += std::string(i ? "," : "") + std::to_string(h.lat_bins[i]);
  out += "],\"lat_ms\":" + h.lat_ns.ms() + ",\"recv_ms\":" + h.recv_ns.ms() + ",\"lock_ms\":" + h.lock_ns.ms() +
         ",\"xfer_ms\":" + h.xfer_ns.ms() + ",\"gaps_over_25\":" + std::to_string(h.gaps_over_25) +
         ",\"max_gap_ms\":" + std::to_string(h.max_gap_ns / 1e6) + ",\"by_frame_mod10\":[";
  for (int i = 0; i < 10; i++) {
    out += std::string(i ? "," : "") + "[" + std::to_string(h.by_frame_mod10_n[i]) + "," + std::to_string(h.by_frame_mod10_late[i]) + "]";
  }
  out += "],\"worst\":[";
  for (int i = 0; i < 5 && h.worst[i].lat_ms > 0; i++) {
    const HcaSend &w = h.worst[i];
    char buf[160];
    snprintf(buf, sizeof(buf), "%s{\"lat\":%.2f,\"recv\":%.2f,\"lock\":%.2f,\"xfer\":%.2f,\"phase\":%.2f,\"frame_mod10\":%u,\"main_busy\":%d}",
             i ? "," : "", w.lat_ms, w.recv_ms, w.lock_ms, w.xfer_ms, w.phase_ms, w.frame_mod10, (int)w.main_busy);
    out += buf;
  }
  out += "]}";
  LOGE("pandad_timing_hca %s", out.c_str());
}

}  // namespace pandad_timing
