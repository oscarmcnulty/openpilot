// Fake panda on /dev/spidev0.0 for running the real pandad offline (LD_PRELOAD).
//
// Intercepts open()/ioctl() on the spidev and implements the panda SPI protocol
// (header -> HACK, data -> DACK + response, checksums), with configurable timing:
//   EMU_IOCTL_CPU_US    busy CPU per ioctl (syscall + DMA setup)           [20]
//   EMU_IOCTL_SLEEP_US  blocking wait per ioctl (DMA completion IRQ)       [60]
//   EMU_SPI_HZ          SPI clock, adds bytes*8/hz to the wait             [50000000]
//   EMU_HACK_US         panda delay before the header ACK is ready         [30]
//   EMU_DACK_US         panda delay before the data ACK/response is ready  [40]
//   EMU_HEALTH_US       extra for the health request (blocking ADC reads)  [100]
//   EMU_CAN_FPS         CAN frames/s the panda receives (all buses)        [2500]
//   EMU_HW_TYPE         panda hw type reported                             [9 = tres]
// Outputs:
//   EMU_HCA_LOG         file: CLOCK_MONOTONIC ns of every HCA_01 (bus 0) the "panda" received
//   EMU_SHM             file: struct emu_shm, the pandad main loop iteration it is in, for the harness
//   EMU_TRACE           file: CSV of every SPI transfer (tid, endpoint, request, start, end ns)

#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/spi/spidev.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "board/health.h"

#define SPI_SYNC 0x5AU
#define SPI_HACK 0x79U
#define SPI_DACK 0x85U
#define SPI_CHECKSUM_START 0xABU

struct emu_shm {
  volatile uint64_t iter;           // pandad main loop iterations seen (one can_recv each)
  volatile uint64_t iter_start_ns;  // start of the current iteration (its can_recv header)
  volatile uint64_t iter_10hz;      // last iteration that read the panda health (10Hz/2Hz state readout)
};

enum state { HEADER, WAIT_HACK, DATA, WAIT_DACK, READ };

static int spi_fd = -1;
static enum state st = HEADER;
static uint8_t endpoint;
static uint16_t tx_len, max_rx_len;
static uint8_t request;
static uint64_t ready_ns, xfer_start_ns;
static uint8_t resp[4096];
static uint16_t resp_len;

static double cpu_us = 20, sleep_us = 60, spi_hz = 50e6, hack_us = 30, dack_us = 40, health_us = 100, can_fps = 2500;
static int hw_type = 9;

static uint64_t rx_frames_owed_ns;  // CAN RX generated up to this time
static uint32_t rx_echoes;          // TX echoes to return
static uint32_t rx_addr = 0x100;
static uint8_t write_rem[64];       // CAN write packet split across bulk writes
static uint32_t write_rem_len, write_rem_need;

static FILE *hca_log, *trace;
static struct emu_shm *shm;
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static double env_d(const char *k, double def) {
  const char *v = getenv(k);
  return v ? atof(v) : def;
}

__attribute__((constructor)) static void init(void) {
  cpu_us = env_d("EMU_IOCTL_CPU_US", cpu_us);
  sleep_us = env_d("EMU_IOCTL_SLEEP_US", sleep_us);
  spi_hz = env_d("EMU_SPI_HZ", spi_hz);
  hack_us = env_d("EMU_HACK_US", hack_us);
  dack_us = env_d("EMU_DACK_US", dack_us);
  health_us = env_d("EMU_HEALTH_US", health_us);
  can_fps = env_d("EMU_CAN_FPS", can_fps);
  hw_type = (int)env_d("EMU_HW_TYPE", hw_type);
  if (getenv("EMU_HCA_LOG")) hca_log = fopen(getenv("EMU_HCA_LOG"), "w");
  if (getenv("EMU_TRACE")) trace = fopen(getenv("EMU_TRACE"), "w");
  if (getenv("EMU_SHM")) {
    int fd = open(getenv("EMU_SHM"), O_RDWR | O_CREAT, 0644);
    if (fd >= 0 && ftruncate(fd, sizeof(struct emu_shm)) == 0) {
      shm = mmap(NULL, sizeof(struct emu_shm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (shm == MAP_FAILED) shm = NULL;
    }
  }
}

__attribute__((destructor)) static void fini(void) {
  if (hca_log) fclose(hca_log);
  if (trace) fclose(trace);
}

// ***** the panda side *****

static uint16_t pack_rx_can(uint8_t *out, uint16_t max_len, uint64_t t) {
  // frames received since the last read, plus echoes of what we sent, whole 14 byte packets only
  uint64_t period_ns = (uint64_t)(1e9 / can_fps);
  if (rx_frames_owed_ns == 0) rx_frames_owed_ns = t;
  uint16_t len = 0;
  while (len + 14U <= max_len) {
    bool echo = rx_echoes > 0;
    if (!echo && (rx_frames_owed_ns + period_ns > t)) break;
    uint8_t *p = &out[len];
    memset(p, 0, 14);
    uint32_t addr = echo ? 0x126U : rx_addr;
    p[0] = (uint8_t)(((echo ? 0U : (rx_addr % 3U)) << 1) | (8U << 4));  // bus, dlc 8
    uint32_t a = (addr << 3) | (echo ? 0x2U : 0U);                      // returned bit
    memcpy(&p[1], &a, 4);
    for (int i = 0; i < 8; i++) p[6 + i] = (uint8_t)(addr + i);
    uint8_t c = 0;
    for (int i = 0; i < 14; i++) c ^= p[i];
    p[5] = c;
    len += 14;
    if (echo) {
      rx_echoes--;
    } else {
      rx_frames_owed_ns += period_ns;
      rx_addr = (rx_addr >= 0x6ffU) ? 0x100U : rx_addr + 7U;
    }
  }
  return len;
}

static void handle_can_write(const uint8_t *d, uint32_t len, uint64_t t) {
  uint32_t pos = 0;
  while (pos < len) {
    uint8_t pkt[64];
    uint32_t pkt_len;
    if (write_rem_len > 0) {
      uint32_t take = write_rem_need - write_rem_len;
      if (take > len - pos) take = len - pos;
      memcpy(&write_rem[write_rem_len], &d[pos], take);
      write_rem_len += take;
      pos += take;
      if (write_rem_len < write_rem_need) return;
      memcpy(pkt, write_rem, write_rem_need);
      pkt_len = write_rem_need;
      write_rem_len = 0;
    } else {
      static const uint8_t dlc_to_len[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64};
      pkt_len = 6U + dlc_to_len[d[pos] >> 4];
      if (pos + pkt_len > len) {
        write_rem_need = pkt_len;
        write_rem_len = len - pos;
        memcpy(write_rem, &d[pos], write_rem_len);
        return;
      }
      memcpy(pkt, &d[pos], pkt_len);
      pos += pkt_len;
    }
    uint32_t a;
    memcpy(&a, &pkt[1], 4);
    uint32_t addr = a >> 3;
    uint8_t bus = (pkt[0] >> 1) & 0x7U;
    rx_echoes++;
    if ((addr == 0x126U) && (bus == 0U) && hca_log) {
      // ns, HCA_01 counter, pandad iteration within the 10Hz cycle (0 = the state readout iteration)
      unsigned phase = shm ? (unsigned)((shm->iter - shm->iter_10hz) % 10U) : 0U;
      fprintf(hca_log, "%llu %u %u\n", (unsigned long long)t, pkt[7] & 0xFU, phase);
      fflush(hca_log);
    }
  }
}

static void process_request(const uint8_t *data, uint64_t t) {
  resp_len = 0;
  request = 0;
  ready_ns = t + (uint64_t)(dack_us * 1000);
  if (endpoint == 0) {
    request = data[0];
    uint16_t length;
    memcpy(&length, &data[5], 2);
    switch (request) {
      case 0xc3:  // uid
        resp_len = 12;
        for (int i = 0; i < 12; i++) resp[i] = (uint8_t)(0xa0 + i);
        break;
      case 0xc1:  // hw type
        resp_len = 1;
        resp[0] = (uint8_t)hw_type;
        break;
      case 0xd2: {  // health, steady state while driving
        struct health_t h;
        memset(&h, 0, sizeof(h));
        h.uptime_pkt = (uint32_t)(t / 1000000000ULL);
        h.voltage_pkt = 13800;
        h.ignition_line_pkt = 1;
        h.controls_allowed_pkt = 0;
        h.car_harness_status_pkt = 1;
        h.safety_mode_pkt = 25;  // volkswagenMlb
        h.power_save_enabled_pkt = 0;
        memcpy(resp, &h, sizeof(h));
        resp_len = sizeof(h);
        ready_ns += (uint64_t)(health_us * 1000);
        break;
      }
      case 0xc2: {  // can health
        can_health_t c;
        memset(&c, 0, sizeof(c));
        c.can_speed = 5000;
        c.can_data_speed = 20000;
        memcpy(resp, &c, sizeof(c));
        resp_len = sizeof(c);
        break;
      }
      case 0xb2:  // fan rpm
        resp_len = 2;
        resp[0] = resp[1] = 0;
        break;
      default:  // writes and anything else: no data
        break;
    }
  } else if (endpoint == 0x81) {
    resp_len = pack_rx_can(resp, max_rx_len, t);
  } else if (endpoint == 3) {
    handle_can_write(data, tx_len, t);
  }

  if (shm && endpoint == 0 && request == 0xd2) {
    shm->iter_10hz = shm->iter;
  }
}

static int do_transfer(struct spi_ioc_transfer *x) {
  uint64_t t0 = now_ns();
  // busy part of the syscall, then the blocking wait for the DMA completion
  while (now_ns() - t0 < (uint64_t)(cpu_us * 1000)) {}
  uint64_t wait_ns = (uint64_t)(sleep_us * 1000 + x->len * 8.0 / spi_hz * 1e9);
  struct timespec ts = {(time_t)(wait_ns / 1000000000ULL), (long)(wait_ns % 1000000000ULL)};
  nanosleep(&ts, NULL);
  uint64_t t = now_ns();

  pthread_mutex_lock(&mtx);
  const uint8_t *tx = (const uint8_t *)(uintptr_t)x->tx_buf;
  uint8_t *rx = (uint8_t *)(uintptr_t)x->rx_buf;
  if (rx) memset(rx, 0, x->len);

  switch (st) {
    case HEADER:
      if (x->len == 7 && tx[0] == SPI_SYNC) {
        endpoint = tx[1];
        memcpy(&tx_len, &tx[2], 2);
        memcpy(&max_rx_len, &tx[4], 2);
        xfer_start_ns = t0;
        ready_ns = t + (uint64_t)(hack_us * 1000);
        st = WAIT_HACK;
        if (shm && endpoint == 0x81) {
          // pandad reads CAN first thing in every main loop iteration (and again only if the read filled a chunk)
          static uint64_t last_can_ns;
          if (t0 - last_can_ns > 2000000ULL) {
            shm->iter++;
            shm->iter_start_ns = t0;
          }
          last_can_ns = t0;
        }
      }
      break;
    case WAIT_HACK:
      if (t >= ready_ns) {
        rx[0] = SPI_HACK;
        st = DATA;
      }
      break;
    case DATA:
      process_request(tx, t);
      st = WAIT_DACK;
      break;
    case WAIT_DACK:
      if (t >= ready_ns && x->len >= 3) {
        rx[0] = SPI_DACK;
        rx[1] = resp_len & 0xFFU;
        rx[2] = resp_len >> 8;
        st = READ;
      }
      break;
    case READ: {
      // rx points 3 bytes into the host buffer; checksum covers DACK, len, data
      memcpy(rx, resp, resp_len);
      uint8_t c = SPI_CHECKSUM_START ^ SPI_DACK ^ (resp_len & 0xFFU) ^ (resp_len >> 8);
      for (int i = 0; i < resp_len; i++) c ^= resp[i];
      rx[resp_len] = c;
      if (trace) {
        fprintf(trace, "%ld,%02x,%02x,%llu,%llu\n", syscall(SYS_gettid), endpoint, request,
                (unsigned long long)xfer_start_ns, (unsigned long long)t);
      }
      st = HEADER;
      break;
    }
  }
  pthread_mutex_unlock(&mtx);
  return (int)x->len;
}

// ***** libc interposition *****

int open(const char *path, int flags, ...) {
  static int (*real_open)(const char *, int, ...) = NULL;
  if (!real_open) real_open = dlsym(RTLD_NEXT, "open");
  va_list ap;
  va_start(ap, flags);
  int mode = (flags & O_CREAT) ? va_arg(ap, int) : 0;
  va_end(ap);
  if (strcmp(path, "/dev/spidev0.0") == 0) {
    spi_fd = real_open("/dev/null", O_RDWR);
    st = HEADER;
    return spi_fd;
  }
  return real_open(path, flags, mode);
}
int open64(const char *path, int flags, ...) __attribute__((alias("open")));

int ioctl(int fd, unsigned long req, ...) {
  static int (*real_ioctl)(int, unsigned long, ...) = NULL;
  if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
  va_list ap;
  va_start(ap, req);
  void *arg = va_arg(ap, void *);
  va_end(ap);
  if (fd == spi_fd && fd >= 0) {
    if (req == SPI_IOC_MESSAGE(1)) return do_transfer((struct spi_ioc_transfer *)arg);
    return 0;  // mode, speed, bits per word
  }
  return real_ioctl(fd, req, arg);
}
