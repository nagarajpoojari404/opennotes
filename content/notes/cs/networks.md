---
title: "Networks"
date: "2026-10-31"
---

The **Open Systems Interconnection (OSI)** model is a conceptual framework standardized by the ISO that partitions network communication into seven distinct abstraction layers, standardizing how data moves from physical hardware up to end-user software applications.


The 7-layer OSI reference architecture. Source: Maccow Creatives / Getty Images

### Layer 1: Physical Layer

Transmits unstructured raw bit streams over a physical transmission medium (electrical voltages, radio frequencies, optical pulses). It defines pinouts, cable specifications, signaling rates, and physical connectors.

- **Protocol Data Unit (PDU):** Bit
- **Hardware:** Hubs, repeaters, network cables (Cat6, fiber optics), transceivers
- **Protocols & Standards:**
	- **Ethernet physical layer (IEEE 802.3 PHY):** 10BASE-T, 100BASE-TX, 1000BASE-T
		- **Wireless PHY standards:** IEEE 802.11a/b/g/n/ac/ax (physical modulation layers)
		- **Serial & Telephony:** RS-232, RS-485, V.35, T1/E1, ISDN (Layer 1 specs)
		- **Optical Transport:** SONET/SDH, OTN (G.709)
		- **Broadband access:** DSL, DOCSIS (PHY specs)

### Layer 2: Data Link Layer

Provides node-to-node data transfer across the same physical segment, handles physical addressing (MAC addresses), framing, error detection (CRC/FCS), and flow control. Often subdivided into **Media Access Control (MAC)** and **Logical Link Control (LLC)**.

- **Protocol Data Unit (PDU):** Frame
- **Hardware:** Layer 2 switches, network interface cards (NICs), bridges
- **Protocols & Standards:**
	- **LAN:** Ethernet (IEEE 802.3 MAC), Wi-Fi (IEEE 802.11 MAC), Token Ring (IEEE 802.5)
		- **WAN & Point-to-Point:** PPP (Point-to-Point Protocol), HDLC, Frame Relay, ATM
		- **Switching & Redundancy:** STP (IEEE 802.1D Spanning Tree Protocol), RSTP, VLAN tagging (IEEE 802.1Q)
		- **Tunneling & Link Layer Control:** L2TP, PPPoE, ARP\* (Address Resolution Protocol operates across Layer 2 and Layer 3 boundaries to map IPs to MACs)

### Layer 3: Network Layer

Handles host-to-host addressing across different networks, path determination, and packet routing. It abstracts network topologies so data can traverse multiple intermediate hops.

- **Protocol Data Unit (PDU):** Packet
- **Hardware:** Routers, Layer 3 switches
- **Protocols & Standards:**
	- **Addressing & Delivery:** IPv4, IPv6
		- **Control & Diagnostics:** ICMP (Internet Control Message Protocol), ICMPv6, IGMP (multicast group management)
		- **Routing Protocols (Interior):** OSPF, IS-IS, RIP, EIGRP
		- **Routing Protocols (Exterior):** BGP (Border Gateway Protocol)
		- **Network Security & Tunneling:** IPsec (AH, ESP), GRE (Generic Routing Encapsulation)

### Layer 4: Transport Layer

Responsible for end-to-end communication between application processes on host devices. It manages multiplexing/demultiplexing via port numbers, segmentation, reassembly, error correction, and flow/congestion control.

- **Protocol Data Unit (PDU):** Segment (TCP) / Datagram (UDP)
- **Protocols & Standards:**
	- **TCP (Transmission Control Protocol):** Connection-oriented, ordered byte-stream delivery with congestion and flow control
		- **UDP (User Datagram Protocol):** Connectionless, lightweight, low-latency datagram delivery without retransmission
		- **SCTP (Stream Control Transmission Protocol):** Multi-stream, message-oriented transport protocol with multi-homing support
		- **QUIC:** UDP-based transport protocol incorporating integrated TLS 1.3 encryption and multiplexing
		- **DCCP (Datagram Congestion Control Protocol):** Unreliable datagram transport with congestion notification

### Layer 5: Session Layer

Establishes, manages, synchronizes, and terminates communication sessions (dialogues) between applications. It controls full-duplex or half-duplex operations and inserts checkpoints for session recovery.

- **Protocol Data Unit (PDU):** Data
- **Protocols & Standards:**
	- **RPC (Remote Procedure Call):** Manages client-server procedural interactions
		- **NetBIOS:** Session establishment between computers on a LAN
		- **PPTP (Point-to-Point Tunneling Protocol):** Encapsulates and manages tunneling sessions
		- **SMPP (Short Message Peer-to-Peer):** Exchanges SMS messages between SMSCs and messaging systems
		- **SIP (Session Initiation Protocol):** Signaling protocol for setting up and tearing down multimedia/VoIP sessions
		- **Sockets API / Named Pipes:** OS abstractions managing local and remote communication sessions

### Layer 6: Presentation Layer

Translates, formats, encrypts, and compresses data so that application-layer entities exchange readable information regardless of internal representations (e.g., character encoding, byte ordering).

- **Protocol Data Unit (PDU):** Data
- **Protocols & Standards:**
	- **Encryption & Security Handshakes:** TLS / SSL (presentation functions: negotiation of cipher suites, encryption/decryption)
		- **Data Representation & Serialization:** ASN.1, XDR (External Data Representation), JSON, Protocol Buffers, MessagePack
		- **Character Encoding:** ASCII, UTF-8, UTF-16, EBCDIC
		- **Media Compression & Encoding:** JPEG, PNG, GIF, MPEG, H.264, MP3

### Layer 7: Application Layer

Directly interacts with software applications to provide end-user network services, resource discovery, authentication, and application messaging.

- **Protocol Data Unit (PDU):** Data
- **Protocols & Standards:**
	- **Web:** HTTP/1.1, HTTP/2, HTTP/3
		- **File Transfer:** FTP, SFTP, TFTP, NFS, SMB/CIFS
		- **Email & Messaging:** SMTP, IMAP, POP3, XMPP
		- **Network Management & Directory:** DNS, DHCP, SNMP, LDAP, NTP (Network Time Protocol)
		- **Remote Shell & Terminal:** SSH, Telnet

### Layer Summary Matrix

| # | Layer | Primary Function | Addressing / Identifier | Key Protocols |
| --- | --- | --- | --- | --- |
| **7** | Application | Network services for user applications | URLs, URIs, FQDNs | HTTP, DNS, SMTP, SSH, DHCP |
| **6** | Presentation | Formatting, encryption, serialization | Data syntax / schemas | TLS/SSL, ASN.1, JSON, XDR |
| **5** | Session | Dialogue control & session lifecycle | Session IDs, RPC handles | SIP, NetBIOS, RPC, SMPP |
| **4** | Transport | End-to-end delivery & port multiplexing | Port numbers (0–65535) | TCP, UDP, QUIC, SCTP |
| **3** | Network | Routing & logical addressing | IP addresses | IPv4, IPv6, ICMP, OSPF, BGP |
| **2** | Data Link | Framing, local MAC addressing & error checks | MAC addresses, VLAN tags | Ethernet, Wi-Fi, PPP, ARP |
| **1** | Physical | Physical transmission of raw bits | Voltage, wavelength, pins | 1000BASE-T, RS-232, 802.11 PHY |
