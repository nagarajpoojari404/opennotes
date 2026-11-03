---
title: "Kubernetes"
date: "2026-11-01"
---

Kubernetes (K8s) is a distributed control system designed around a single architectural axiom: **declarative state management executed via continuous reconciliation loops.** Instead of executing imperative run commands, you declare a target topology, and an array of autonomous, decoupled controllers constantly work to converge the cluster's actual state to that declared state.

### 1\. Control Plane Architecture: The State & Decision Engine

The control plane coordinates cluster logic, enforces policies, and holds persistent state.

```
+-------------------------------------------------------+
                |                    Control Plane                      |
                |                                                       |
  kubectl ----> |  [kube-apiserver] <-------> [etcd (Raft Quorum)]      |
                |         ^                                             |
                |         |-- [kube-controller-manager] (Reconciliation)|
                |         |-- [kube-scheduler] (Filter -> Score)        |
                +---------|---------------------------------------------+
                          | Watch API (HTTP/2 gRPC Streams)
       +------------------+------------------+
       |                                     |
       v                                     v
+--------------------------+     +--------------------------+
|       Worker Node 1      |     |       Worker Node 2      |
| [kubelet] -> CRI runtime |     | [kubelet] -> CRI runtime |
| [kube-proxy] (iptables/  |     | [kube-proxy] (iptables/  |
|               IPVS/eBPF) |     |               IPVS/eBPF) |
| Pod (veth / netns)       |     | Pod (veth / netns)       |
+--------------------------+     +--------------------------+
```

#### `etcd`: The Canonical Source of Truth

- **Mechanism:** A consistent, distributed key-value store using the **Raft consensus algorithm**.
- **Storage Schema:** Keys are stored hierarchically (e.g., `/registry/pods/<namespace>/<pod-name>`). It maintains a monotonically increasing 64-bit counter called a `ResourceVersion` (backed by Raft commit indices and bbolt b-trees).
- **Isolation:** Only `kube-apiserver` communicates directly with `etcd`. No other component (kubelet, scheduler, etc.) ever reads or writes to `etcd` directly.

#### `kube-apiserver`: The Gateway & State Serializer

The API server is stateless, scales horizontally behind a layer-4 load balancer, and exposes a RESTful JSON/Protobuf HTTP/2 API. When an incoming payload arrives, it runs through a strict linear pipeline:

1. **Authentication:** Decodes bearer tokens, X.509 client certs, or OIDC claims to establish `User`, `Groups`, and `Extra`.
2. **Authorization:** Checks privileges via **RBAC** (Role-Based Access Control), ABAC, or Webhook authorizers.
3. **Mutating Admission Webhooks:** Invokes webhooks to modify payloads (e.g., injecting sidecars or default resource requests).
4. **Schema Validation:** Verifies syntax, required fields, and structural specifications against the OpenAPI schema.
5. **Validating Admission Webhooks:** Runs policy validation (e.g., OPA Gatekeeper, Kyverno) to reject non-compliant specs.
6. **Persistence:** Commits to `etcd` and assigns a new `ResourceVersion`.
7. **Watch Distribution:** Streams changes via long-lived HTTP/2 streams (`watch`) to listeners using the `Reflector` mechanism inside client-go.

#### `kube-controller-manager`: The Reconciliation Engine

Runs dozens of discrete control loops inside a single process. A controller implements the **level-triggered reconciliation loop**:

INI8 Labs

$$
\text{Reconcile}() : \quad \text{Actual State} \xrightarrow{\quad\Delta\quad} \text{Desired State}
$$

Using client-go **Informer** and **Lister** patterns, controllers cache state locally to avoid overloading `kube-apiserver`. When an event occurs, an informer pushes the key into a `WorkQueue`, and worker threads pop the key, calculate delta, and post patch requests back to the API server.

- *DeploymentController* reconciles `Deployment` \\rightarrow creates/updates `ReplicaSet`.
	INI8 Labs
- *ReplicaSetController* reconciles `ReplicaSet` $\rightarrow$ creates/updates raw `Pod` manifests.
- *NodeController* tracks node heartbeats (`Leases`) and evicts pods when nodes become `NotReady`.
	INI8 Labs

#### `kube-scheduler`: Algorithmic Placement

Monitors pods where `spec.nodeName == ""`. Placement runs in a two-phase cycle:

1. **Filtering (Predicates):** Prunes ineligible nodes (e.g., checks resource requests vs. node allocatable capacity, node selectors, taints/tolerations, port collisions, volume zone topology).
2. **Scoring (Priorities):** Assigns a 0–100 score across remaining nodes based on weighted rules (e.g., image locality, inter-pod anti-affinity spreading, balanced resource allocation).
3. **Binding:** Atomically updates the pod's `spec.nodeName` via a `/binding` subresource call to `kube-apiserver`.

### 2\. Worker Node Architecture: The Execution Engine

Worker nodes run the scheduled containers and manage the low-level operating system primitives.

DEV Community

#### `kubelet`: Node Daemon

A systemd service acting as the cluster's node agent. It watches the API server for pods assigned to its node.

Medium

- **SyncLoop:** kubelet's core event loop multiplexes changes from three channels: file updates, HTTP endpoints, and the API server watch stream.
- **PLEG (Pod Lifecycle Event Generator):** Periodically polls the container runtime and translates raw OS process states into unified Pod lifecycle events.
- **CAdvisor:** Gathers node- and container-level resource statistics (CPU, memory, network, disk) and exposes them to metrics endpoints.

#### Container Runtime Interface (CRI)

kubelet does not manage Linux namespaces or cgroups directly; it communicates over a local Unix domain socket using gRPC with a CRI-compliant runtime (like `containerd` or `CRI-O`). CRI breaks into two distinct services:

- **ImageService:** Pulls images, unpacks root filesystems, and manages storage layers.
	Medium
- **RuntimeService:** Handles container and pod sandbox lifecycles.
	Medium

Under the hood:

1. `containerd` calls an OCI runtime (e.g., `runc`, `crun`).
2. The runtime configures Linux kernel primitives:
	- **Namespaces:** `pid`, `net`, `ipc`, `mnt`, `uts`, `user` to isolate system resources.
		- **Cgroups (v2):** Restricts memory limits, CFS (Completely Fair Scheduler) CPU quotas, and I/O pressure.
		- **Pivot\_root / OverlayFS:** Mounts the container's isolated root filesystem layers.

#### Pod Sandbox & The `pause` Container

Every pod begins by launching an infrastructure container (the **`pause` container**). The `pause` container makes a `clone()` syscall with new network and IPC namespace flags, and then sleeps indefinitely via `pause()`. All application containers in that pod join the `pause` container's network namespace via `setns()`. This allows all containers within the pod to communicate over `localhost` and share the same IP address.

VMware

### 3\. Networking Internals

Kubernetes enforces three invariants:

1. Every pod gets its own unique, routable IP address.
	VMware
2. Pods can communicate with all other pods across nodes without NAT.
3. Agents on a host can communicate with all pods on that host.

#### Container Network Interface (CNI)

When a pod sandbox is created, kubelet delegates network creation to the CNI plugin (e.g., Cilium, Calico, Flannel).

1. The CNI plugin creates a **Virtual Ethernet pair (`veth`)**.
2. It pushes one end into the pod's network namespace (renamed to `eth0`) and anchors the other end in the host root network namespace (e.g., `veth1234`).
3. It assigns an IP address to the pod's `eth0` via an IPAM (IP Address Management) plugin.
	VMware
4. It sets up cross-node routing using either **Overlay networks** (VXLAN/Geneve tunneling encapsulation) or **Direct Routing / BGP** (programming the host route tables or upstream top-of-rack switches).

#### `kube-proxy` & Service Abstraction

A `Service` is not a process; it is a static Virtual IP (ClusterIP) fronting dynamic Pod IPs. `kube-proxy` watches Service and Endpoints/EndpointSlice objects, implementing traffic forwarding using one of three backends:

Medium - Medium

| Proxy Mode | Implementation Mechanism | Characteristics |
| --- | --- | --- |
| **iptables** | Generates chains inside Netfilter `nat` tables using PREROUTING and OUTPUT hooks. | Random probability-based matching (`-m statistic --mode random`). $O(n)$ sequential chain evaluation causes latency spikes at thousands of services. |
| **IPVS** | Kernel-space transport-layer load balancer utilizing hash tables inside Netfilter hooks. | $O(1)$ lookup performance, supports round-robin, least-connection, and session-affinity balancing algorithms. |
| **eBPF (Cilium)** | Replaces kube-proxy entirely by attaching BPF programs to `tc` (Traffic Control) or socket layers (`sock_ops`). | Bypasses core Netfilter connection tracking (`conntrack`), translating service VIPs to pod endpoints directly at the socket level. |

### 4\. End-to-End Walkthrough: `kubectl apply -f deployment.yaml`

To see how all components coordinate in real time, trace the lifecycle of a Deployment:

1. **Submission:** `kubectl` sends an HTTP POST with the Deployment manifest to `kube-apiserver`.
2. **Admission & Persistence:** `kube-apiserver` authenticates the token, checks RBAC permissions, runs mutating webhooks, validates the spec, writes the object to `etcd`, and acknowledges the client.
3. **Deployment Controller:** The `DeploymentController` detects the new resource via its watch cache and creates a corresponding `ReplicaSet`.
4. **ReplicaSet Controller:** The `ReplicaSetController` notices the missing replicas and writes new `Pod` specs with empty `spec.nodeName` fields.
5. **Scheduling:** `kube-scheduler` filters out invalid nodes, scores remaining candidates, and executes a binding call setting `spec.nodeName = "node-1"`.
6. **Execution (Kubelet):** The `kubelet` on `node-1` picks up the binding event via its watch stream:
	- Calls the container runtime via CRI to spawn the `pause` container.
		- Invokes the CNI plugin to configure interfaces, routing, and IP assignments.
		- Pulls application container images, sets cgroups and namespace restrictions, and starts the workload.
		- Reports container status back to the API server.
7. **Service Ingress:** If a Service exposes the Deployment, `kube-proxy` on every node detects the pod's new IP and updates the local routing tables (iptables/IPVS/eBPF maps).
