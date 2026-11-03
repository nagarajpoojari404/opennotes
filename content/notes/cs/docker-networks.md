---
title: "Docker networks"
date: "2026-11-01"
---

Docker provides several built-in network drivers to handle container communication, isolation, and routing. Each driver serves a specific architectural use case depending on whether containers run on a single host, across multiple hosts, or require direct physical network integration.

### 1\. Bridge Network (Default)

The bridge network creates a software switch on the host (usually named `docker0`). Containers attached to the same bridge can communicate with each other via internal IP addresses.

- **Default Bridge:** Automatically assigned to containers if no network is specified. Does not support automatic DNS resolution by container name; containers must communicate via IP or manual `--link`.
- **User-Defined Bridge:** Custom bridge created with `docker network create -d bridge <name>`. Provides automatic service discovery (DNS resolution via container names) and better isolation.
- **Best used for:** Standalone containers running on a single host needing private, isolated inter-container communication.
Bash

```
# Create and run on a user-defined bridge
docker network create my-bridge
docker run -d --name web --network my-bridge nginx
docker run -it --network my-bridge alpine ping web
```

### 2\. Host Network

The host driver removes network isolation between the container and the Docker host. The container shares the host’s networking namespace directly.

- **Port Behavior:** Port mapping (`-p` or `-P`) is ignored. If a service in the container binds to port `80`, it immediately binds to port `80` on the host machine.
- **Performance:** Eliminates Network Address Translation (NAT) overhead, achieving near-native network throughput.
- **Limitations:** Only works natively on Linux hosts (on macOS and Windows, Docker runs inside a lightweight VM, so the host network attaches to the VM rather than your actual OS).
- **Best used for:** High-throughput, low-latency applications (e.g., media streaming, high-frequency trading) or services monitoring the host’s network stack.
Bash

```
docker run -d --name cache --network host redis
```

### 3\. None Network

The `none` driver disables all networking for the container.

- **Configuration:** The container only receives a loopback interface (`127.0.0.1` / `::1`). No external interfaces, gateway, or routing table are created.
- **Best used for:** Completely air-gapped tasks, batch jobs running untrusted code, or containers generating local cryptographic keys where external connectivity poses a security risk.
Bash

```
docker run -it --network none alpine ifconfig
```

### 4\. Overlay Network

The overlay driver manages multi-host networking by encapsulating Layer 2 Ethernet frames inside Layer 4 UDP packets (VXLAN tunnels) across different Docker daemon hosts.

- **Scope:** Requires Docker Swarm mode or an external key-value store (like Consul/etcd) to sync routing tables across nodes.
- **DNS & Routing:** Provides automatic multi-host service discovery and ingress routing (routing mesh), distributing traffic across Swarm replicas automatically.
- **Best used for:** Multi-host orchestration, Swarm clusters, and distributed container systems without third-party network plugins.
Bash

```
# Swarm must be initialized first: docker swarm init
docker network create -d overlay --attachable my-overlay
```

### 5\. Macvlan Network

The macvlan driver assigns a distinct physical MAC address to each container, making them appear as physical devices directly attached to the host's local network router.

- **Subnet Integration:** Containers obtain routable IP addresses directly from the local LAN’s DHCP server or a dedicated subnet.
- **Host Caveat:** By default, Linux kernel routing prevents communication directly between the Docker host and its own macvlan containers.
- **Best used for:** Legacy enterprise applications requiring direct Layer 2 connectivity, network monitoring tools, or setups where external network firewalls inspect container traffic by IP/MAC address.
Bash

```
docker network create -d macvlan \
  --subnet=192.168.1.0/24 \
  --gateway=192.168.1.1 \
  -o parent=eth0 my-macvlan
```

### 6\. Ipvlan Network

Similar to macvlan, the ipvlan driver provides direct Layer 2/Layer 3 network integration, but all containers share the parent interface’s physical MAC address while maintaining distinct IP addresses.

- **Modes:**
	- **L2 Mode:** Operates like a switch; containers share the host MAC but have individual IPs. Requires standard ARP resolution.
		- **L3 Mode:** Operates as a router; eliminates broadcast traffic, ARP, and Layer 2 switching. The Docker host routes traffic between subnets.
- **Best used for:** Networks with strict MAC address limits (e.g., public cloud hypervisors, switches with port security enabled) or high-density container deployments where ARP storms must be avoided.
Bash

```
docker network create -d ipvlan \
  --subnet=192.168.20.0/24 \
  --gateway=192.168.20.1 \
  -o ipvlan_mode=l2 \
  -o parent=eth0 my-ipvlan
```

### 7\. Network Plugins (Third-Party / CNI)

Docker supports third-party networking extensions via the Container Network Interface (CNI) or custom Docker network plugins.

- **Examples:** Calico, Weave Net, Cilium, Flannel.
- **Features:** Advanced capabilities such as BGP routing, eBPF data planes, cross-cluster wireguard encryption, and fine-grained network policies/firewalls.
- **Best used for:** Production Kubernetes migrations, regulated environments needing micro-segmentation, and cloud-native service meshes.

### Driver Comparison

| Driver | Host Scope | Multi-Host? | Performance | Automatic DNS? | Common Use Case |
| --- | --- | --- | --- | --- | --- |
| **Bridge** | Single | No | Good (NAT overhead) | Yes (User-defined only) | Standard web stacks, local microservices |
| **Host** | Single | No | Maximum (Native) | Host-dependent | High performance, raw port binding |
| **None** | Single | No | N/A | None | Air-gapped batch jobs, security sandboxing |
| **Overlay** | Multi | Yes | Moderate (VXLAN encapsulation) | Yes | Docker Swarm, multi-node setups |
| **Macvlan** | Single/LAN | Direct to LAN | High | Router-dependent | Legacy migrations, VLAN compliance |
| **Ipvlan** | Single/LAN | Direct to LAN | High | Router-dependent | High-density networks, MAC-restricted switches |
