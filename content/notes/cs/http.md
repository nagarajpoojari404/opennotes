---
title: "HTTP"
date: "2026-10-31"
---

The evolution across **HTTP/1.1**, **HTTP/2**, and **HTTP/3** reflects a continuous push to eliminate latency, reduce network round-trips, and overcome transmission bottlenecks across the web.

### HTTP/1.1 (1997 / 1999) — The Text-Based Baseline

HTTP/1.1 standardized modern web semantics (verbs like `GET`, `POST`, status codes, headers, and caching directives) and introduced key persistence improvements over HTTP/1.0.

- **Underlying Transport:** Plain TCP (Layer 4).
- **Format:** Human-readable ASCII text headers and messages separated by CRLF (`\r\n`).
- **Key Innovations:**
	- **Persistent Connections (`Keep-Alive` by default):** Allows multiple request/response cycles over a single TCP connection, avoiding repeated 3-way TCP handshakes.
		- **Chunked Transfer Encoding:** Allows streaming responses of unknown initial payload size using chunk headers.
		- **Pipelining:** Clients could send multiple requests back-to-back without waiting for responses, though rarely deployed due to intermediary proxy bugs.
- **The Core Bottleneck — Application Head-of-Line (HoL) Blocking:**
	- HTTP/1.1 enforces strict FIFO ordering per TCP connection. If Request #1 stalls on the server (e.g., waiting on a slow database query), Request #2 and #3 queued behind it must wait, even if ready to deliver.
		- Browsers worked around this by opening 6 parallel TCP connections per host domain, introducing significant socket, handshake, and TLS negotiation overhead.

### HTTP/2 (2015) — Multiplexing Over a Single Connection

HTTP/2 retained HTTP/1.1 request/response semantics (`GET`, `POST`, headers, status codes) while completely overhauling the wire format and framing mechanism to eliminate application-layer HoL blocking.

- **Underlying Transport:** TCP + TLS (Layer 4). In practice, HTTP/2 is almost universally deployed over encrypted TLS (h2).
- **Format:** Strict Binary Framing Layer. Every message is split into typed binary frames (`HEADERS`, `DATA`, `SETTINGS`, `RST_STREAM`, etc.).
- **Key Innovations:**
	- **True Stream Multiplexing:** Multiple bidirectional streams interleave independently over a **single TCP connection**. Packets from Stream 1, Stream 3, and Stream 5 can be sent concurrently without waiting for Stream 1 to complete.
		- **HPACK Header Compression:** HTTP headers are statefully compressed via pre-indexed static tables and dynamically synchronized differential tables between client and server, dramatically reducing redundant header overhead across requests.
		- **Stream Prioritization & Flow Control:** Allows the client to signal resource weight/dependency (e.g., render-blocking CSS prioritized over footer images).
		- **Server Push:** Allowed servers to pre-emptively push assets to the client cache (though deprecated in modern browsers due to cache invalidation complexity).
- **The Remaining Bottleneck — TCP-Level Head-of-Line Blocking:**
	- Because all logical streams share one OS-level TCP socket, TCP's byte-stream guarantees enforce strict packet order. If a single IP packet drops, the OS TCP receiver buffer freezes **all** HTTP/2 streams on that connection until the lost segment is retransmitted. Under high packet loss (>2%), HTTP/2 can perform worse than multi-connection HTTP/1.1.

### HTTP/3 (2022) — Transport Reinvention Over QUIC

HTTP/3 completely decouples HTTP from traditional TCP, replacing TCP and TLS with **QUIC** (a user-space, transport-layer protocol running on top of UDP).

- **Underlying Transport:** **QUIC over UDP** (Layer 4).
- **Format:** Binary frames mapped directly to QUIC stream primitives.
- **Key Innovations:**
	- **Zero Transport HoL Blocking:** QUIC implements streams natively in transport logic. If a packet belonging to Stream A drops on the wire, only Stream A pauses. Stream B, Stream C, and Stream D continue streaming to the application without interruption.
		- **Integrated TLS 1.3 Encryption:** Encryption is baked directly into QUIC's transport layer rather than sitting on top. Handshake round-trips drop to **1-RTT** on first connection and **0-RTT** on resumption (data sent alongside the initial connection frame).
		- **Connection Migration via Connection IDs:** TCP connections are bound to a 4-tuple (`Src IP`, `Src Port`, `Dst IP`, `Dst Port`). Switching from Wi-Fi to cellular breaks TCP connections. QUIC uses a random 64-bit Connection ID independent of IP/port, allowing persistent connections to survive network network handoffs seamlessly without re-handshaking.
		- **QPACK Header Compression:** A redesign of HPACK that handles unordered packet arrivals across streams without causing transport stalls.

### Structural Comparison

| Feature | HTTP/1.1 | HTTP/2 | HTTP/3 |
| --- | --- | --- | --- |
| **Underlying Layer 4** | TCP | TCP | QUIC (over UDP) |
| **Framing Format** | Plain ASCII Text | Binary Frames | Binary Frames (via QUIC) |
| **Multiplexing** | No (FIFO per connection) | Yes (Application-layer) | Yes (Transport-native) |
| **Head-of-Line Blocking** | At Application layer | At TCP transport layer | None |
| **Header Compression** | None | HPACK | QPACK |
| **Handshake Latency** | 2–3 RTTs (TCP + TLS) | 2–3 RTTs (TCP + TLS) | 0–1 RTT (QUIC + TLS 1.3) |
| **Connection Migration** | Fails on IP/network change | Fails on IP/network change | Native via Connection ID |
