---
title: "Latency numbers"
date: "2026-11-01"
---

The classic list originally popularized by **Jeff Dean** (and later expanded by Peter Norvig and Colin Scott) provides essential orders of magnitude for systems engineering and performance estimation.

GitHub Gist

### 1\. Canonical Latency Numbers

| Operation | Latency | Standard Representation |
| --- | --- | --- |
| **L1 cache reference** | 0.5 – 1 ns | ~1 ns |
| **Branch mispredict** | 3 – 5 ns | ~3–5 ns |
| **L2 cache reference** | 3 – 7 ns | ~4–7 ns |
| **Mutex lock / unlock** | 15 – 25 ns | ~20 ns |
| **Main memory reference (RAM)** | 50 – 100 ns | ~100 ns |
| **Compress 1 KB with Snappy/Zippy** | 2,000 – 3,000 ns | ~2–3 µs |
| **Send 2 KB over 10 Gbps network** | ~1,600 ns | ~1.6 µs |
| **Send 1 KB over 1 Gbps network** | 10,000 ns | 10 µs |
| **SSD random read (4 KB, NVMe/PCIe)** | 10,000 – 100,000 ns | 10 – 100 µs |
| **Read 1 MB sequentially from memory** | 3,000 – 10,000 ns *(modern)*      250,000 ns *(classic)* | 10 – 250 µs |
| **Round trip in same datacenter** | 500,000 ns | 0.5 ms |
| **Read 1 MB sequentially from SSD** | 200,000 – 1,000,000 ns | 0.2 – 1 ms |
| **HDD seek time (rotational latency + arm)** | 2,000,000 – 10,000,000 ns | 2 – 10 ms |
| **Read 1 MB sequentially from HDD** | 5,000,000 – 20,000,000 ns | 5 – 20 ms |
| **Cross-continental packet round trip (e.g., US to Europe)** | 150,000,000 ns | 150 ms |

### 2\. Scaled to Human Time (1 CPU Cycle ≈ 1 Second)

Scaling **0.5–1 ns to ~1 second** highlights the stark gaps across the hardware hierarchy:

| Operation | Actual Time | Scaled Time | Real-World Analogy |
| --- | --- | --- | --- |
| **L1 cache hit** | 0.5 – 1 ns | **1 second** | Grabbing a piece of paper on your desk |
| **L2 cache hit** | ~4 ns | **4 seconds** | Grabbing a book from a shelf within reach |
| **RAM access** | ~100 ns | **1.5 minutes** | Walking down the hall to fetch a file |
| **NVMe SSD random read** | ~20 µs | **~5.5 hours** | A half-day road trip |
| **Same-datacenter network RTT** | ~500 µs | **~5.8 days** | Taking a week off |
| **HDD seek** | ~10 ms | **~4 months** | A semester abroad |
| **Transatlantic packet RTT** | ~150 ms | **~5 years** | Completing an entire degree |

### 3\. Rules of Thumb for System Design

1. **Avoid Random Disk I/O**: Sequential throughput on disk and flash is orders of magnitude faster than random seeks.
	Google SRE
2. **Datacenter vs. WAN Boundaries**: In-rack or same-datacenter communication (~0.5 ms) allows ~2,000 round trips/sec, whereas cross-continent WAN round trips (~150 ms) max out at 6–7 trips/sec.
	Google SRE
3. **RAM Is Fast, but Cache Locality Still Dominates**: A single cache miss that hits main memory takes ~100× longer than an L1 cache hit. Data-oriented layout (contiguous arrays, avoiding pointer chasing) maximizes hardware prefetching and cache line utilization.
