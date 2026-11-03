---
title: "Kubernetes objects"
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

```
+---------+         +----------------+         +------+         +--------------------------+         +----------------+         +---------+
| kubectl |         | kube-apiserver |         | etcd |         | kube-controller-manager  |         | kube-scheduler |         | kubelet |
+---------+         +----------------+         +------+         +--------------------------+         +----------------+         +---------+
     |                      |                     |                           |                             |                        |
     |--- POST /deploy ---->|                     |                           |                             |                        |
     |    (AuthN/AuthZ/Webhooks)                  |                           |                             |                        |
     |                      |--- Write /depl ---->|                           |                             |                        |
     |                      |<-- Commit ACK ------|                           |                             |                        |
     |                      |                                                 |                             |                        |
     |                      |=== Watch (Deployment Created) =================>|                             |                        |
     |                      |                                                 |-- Reconcile: creates RS     |                        |
     |                      |<-- POST /replicasets ---------------------------|                             |                        |
     |                      |=== Watch (ReplicaSet Created) ================>|                             |                        |
     |                      |                                                 |-- Reconcile: creates Pods   |                        |
     |                      |<-- POST /pods (spec.nodeName="") ---------------|                             |                        |
     |                      |                                                                               |                        |
     |                      |=== Watch (Unscheduled Pods) =================================================>|                        |
     |                      |                                                                               |-- Filter -> Score      |
     |                      |<-- POST /pods/binding (node="node-1") ----------------------------------------|                        |
     |                      |                                                                                                        |
     |                      |=== Watch (Pod scheduled on node-1) ===================================================================>|
     |                      |                                                                                                        |-- Calls CRI (sandbox, pause)
     |                      |                                                                                                        |-- Calls CNI (veth, IP)
     |                      |                                                                                                        |-- Pulls image & starts container
     |                      |<-- PATCH /pods/status (Phase="Running") ----------------------------------------------------------------|
     v                      v                                                                                                        v
```
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

Want to dive deep into how eBPF bypasses iptables in modern Kubernetes networking?

Yes

##### You said now list and explain all kubernetes objects

now list and explain all kubernetes objects

###### Gemini said

In Kubernetes, an **Object** is a persistent entity in `etcd` that represents the intent of the cluster (its desired state) alongside its current status. Objects belong to specific API Groups and are split by scope: **Namespaced** (isolated to an environment) or **Cluster-scoped** (global infrastructure).

### 1\. Workload Objects (Compute & Lifecycles)

These objects dictate how, where, and for how long your containerized workloads execute.

Kubernetes

- **Pod:** The atomic deployable unit in Kubernetes. A wrapper around one or more tightly coupled containers that share IPC, Linux namespaces, and an IP address via a shared network namespace.
	Kubernetes
- **ReplicaSet:** Ensures a specified number of identical Pod replicas remain running at all times. Uses set-based label selectors to acquire or evict running Pods.
- **Deployment:** A declarative supervisor over ReplicaSets. Manages rolling updates, rollbacks, and zero-downtime version transitions for stateless workloads.
- **StatefulSet:** Manages stateful applications (e.g., databases, distributed stores). Provides stable, persistent network identifiers (e.g., `db-0`, `db-1`), ordered deployment/termination, and automated provisioning of dedicated storage volumes per replica.
	Kubernetes\+ 1
- **DaemonSet:** Guarantees that a copy of a specific Pod runs on all (or targeted) nodes. Commonly used for host log collectors, metrics scrapers, and storage or CNI network daemons.
	Kubernetes
- **Job:** Supervises batch processes designed to run to completion rather than stay up continuously. Retries failing pods until a specified number of successful completions is reached.
	Kubernetes
- **CronJob:** A time-based scheduler (cron syntax) that instantiates `Job` objects at designated intervals or deadlines.

### 2\. Discovery, Routing & Networking Objects

These resources stitch dynamic, ephemeral pod IPs together into static network endpoints and handle inbound ingress.

Kubernetes

- **Service:** An abstraction defining a logical set of Pods and a consistent policy to access them.
	- *ClusterIP:* Virtual IP accessible only internally inside the cluster.
		- *NodePort:* Exposes the Service on a static port across every node's external IP.
		- *LoadBalancer:* Provisions an external cloud-provider load balancer routing to the NodePort/Pods.
		- *Headless (`clusterIP: None`):* Returns direct Pod IPs via DNS A/AAAA records for direct peer discovery.
- **Endpoints / EndpointSlice:** The dynamic address book behind a Service. Contains the actual IP and port tuples of healthy Pods matching the Service’s selector. `EndpointSlice` splits large lists into chunks (default 100 endpoints) to avoid control plane serialization bottlenecks.
- **Ingress:** Layer 7 (HTTP/HTTPS) routing rules (host, path, TLS termination) mapping inbound external requests to internal Services. Implemented by an Ingress Controller (e.g., NGINX, Envoy).
- **Gateway API (`GatewayClass`, `Gateway`, `HTTPRoute`, `GRPCRoute`):** The modern, expressive, role-oriented successor to Ingress. Decouples infrastructure provisioning (cluster ops) from route matching and traffic splitting (app developers).
- **NetworkPolicy:** Firewall rules enforced at the CNI layer. Dictates ingress and egress packet filtering for Pods based on IP CIDRs, namespaces, and pod labels.

### 3\. Configuration & Secrets Objects

Used to inject operational variables, configs, and credentials without rebuilding container images.

- **ConfigMap:** Key-value store for non-confidential configuration data (environment variables, config files, command-line arguments) mounted as files or injected as environment variables.
- **Secret:** Stores sensitive data (keys, certificates, passwords) encoded in Base64. Can be mounted as memory-backed `tmpfs` volumes to prevent secrets from touching node disks.

### 4\. Storage Objects

Abstracts physical disk volumes (cloud block stores, NFS, local NVMe) into portable cluster assets.

- **PersistentVolume (PV):** A piece of networked storage in the cluster provisioned by an administrator or dynamically through a StorageClass. Cluster-scoped.
- **PersistentVolumeClaim (PVC):** A request for storage by a user (specifying access modes, size, and StorageClass). Namespaced. Bound 1:1 with an appropriate PV.
- **StorageClass:** Defines storage dynamic provisioning parameters (provisioner plugin, volume type, IOPS tier, reclaim policies such as `Delete` or `Retain`).
- **VolumeAttachment:** Tracks the attaching of a CSI (Container Storage Interface) volume to a specific worker node.
- **CSIDriver & CSINode:** Cluster-level metadata specifying installed third-party storage drivers and node-specific discovery details.

### 5\. Cluster Topology & Administrative Objects

Resources that partition the cluster, describe machines, and coordinate leases.

- **Namespace:** Virtual clusters backed by a single physical cluster, providing scoping for object names, RBAC boundaries, and resource quotas.
- **Node:** Represents a physical machine or VM in the cluster, registering allocatable CPU/memory, kernel version, condition heartbeats, and taints.
- **Lease:** Lightweight coordination primitives inside the `coordination.k8s.io` group. Used by control plane components for leader election and by Kubelets to publish node health heartbeats.
- **ResourceQuota:** Sets hard upper bounds on total resource consumption (CPU, memory, object counts) within a single Namespace.
- **LimitRange:** Enforces min/max resource constraints and default resource requests/limits for individual containers inside a Namespace.

### 6\. Security, Identity & Access Control (RBAC)

Objects governing authentication, authorization, and process-level privilege escalation.

- **ServiceAccount:** Identity assigned to processes running inside Pods to authenticate against the `kube-apiserver`.
- **Role / ClusterRole:** Sets of additive permissions (`get`, `list`, `watch`, `create`, etc.) on API resources. `Role` is namespaced; `ClusterRole` applies globally across all namespaces.
- **RoleBinding / ClusterRoleBinding:** Grants the permissions defined in a Role/ClusterRole to a subject (User, Group, or ServiceAccount).
- **PodDisruptionBudget (PDB):** Defines the minimum available or maximum unavailable replicas during voluntary disruptions (e.g., node drain, maintenance updates).
- **CertificateSigningRequest (CSR):** Used to request X.509 client/server certificates signed by the cluster's internal Certificate Authority (CA).

### 7\. Extensibility & Dynamic Admission

Objects that allow extending the Kubernetes API schema and intercepting requests at runtime.

- **CustomResourceDefinition (CRD):** Extends the Kubernetes API by defining new custom object kinds, schemas (OpenAPI v3), and endpoints without modifying core code.
	Kubernetes
- **MutatingWebhookConfiguration:** Directs `kube-apiserver` to call external HTTP services to modify objects prior to persistence (e.g., auto-injecting monitoring agents).
- **ValidatingWebhookConfiguration:** Directs `kube-apiserver` to evaluate objects against external admission webhooks, accepting or rejecting payloads based on organization policy.

### Summary Reference Table

| Functional Domain | Primary Kubernetes Objects | Scope |
| --- | --- | --- |
| **Workloads** | `Pod`, `Deployment`, `ReplicaSet`, `StatefulSet`, `DaemonSet`, `Job`, `CronJob` | Namespaced |
| **Networking** | `Service`, `Endpoints`, `EndpointSlice`, `Ingress`, `NetworkPolicy`, `HTTPRoute` | Namespaced |
| **Configuration** | `ConfigMap`, `Secret` | Namespaced |
| **Storage (User)** | `PersistentVolumeClaim` | Namespaced |
| **Storage (Infra)** | `PersistentVolume`, `StorageClass`, `CSIDriver`, `VolumeAttachment` | Cluster |
| **Cluster & Identity** | `Namespace`, `Node`, `ServiceAccount`, `Role`, `RoleBinding`, `ClusterRole`, `ClusterRoleBinding` | Mixed |
| **Governance & Limits** | `ResourceQuota`, `LimitRange`, `PodDisruptionBudget` | Namespaced |
| **Extensions** | `CustomResourceDefinition`, `MutatingWebhookConfiguration`, `ValidatingWebhookConfiguration` | Cluster |
