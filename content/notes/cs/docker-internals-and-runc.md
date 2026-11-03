---
title: "Docker internals and runc"
date: "2026-10-31"
---

At the Linux kernel level, there is no discrete entity called a "container." A container is simply a standard Linux process isolated by **namespaces**, resource-constrained by **cgroups**, bound to a dedicated root filesystem using **chroot/pivot\_root**, and restricted by security mechanisms like **seccomp** and **capabilities**.

### The Runtime Architecture: Docker to runc

Running `docker run` triggers a chain of components adhering to Open Container Initiative (OCI) standards:

1. **Docker CLI / Engine (`dockerd`)**: Accepts the client request, manages local storage, interacts with image registries, unpacks image layers, and calls `containerd` over a gRPC API.
2. **`containerd`**: Manages the container lifecycle, image distribution, and snapshot storage. When told to create a container, it delegates the low-level execution to a runtime shim (`containerd-shim`).
3. **`containerd-shim`**: Sits between `containerd` and the low-level runtime. It holds the standard I/O (stdin/stdout/stderr) pipes and file descriptors open, reports exit statuses, and prevents the container from dying if `dockerd` or `containerd` crashes or restarts.
4. **`runc`**: The OCI reference implementation. It reads an OCI bundle (a directory containing a `rootfs` filesystem and a JSON configuration file called `config.json`), makes the low-level Linux syscalls to configure the isolation environment, launches the containerized process, and immediately exits, leaving the process running under the shim.

### The Core Linux Primitives

#### 1\. Linux Namespaces (What the Process Can See)

Namespaces partition kernel resources so that a process inside a namespace sees its own isolated instance of global resources. When `runc` initializes a container, it uses the `clone()` system call with specific flags, or joins existing namespaces using `unshare()` and `setns()`.

- **PID (Process IDs)**: The container process becomes PID 1 inside its own namespace, handling signals and child reaping, while retaining a regular PID on the host.
- **NET (Network Devices & Routing)**: Isolates network interfaces, IP routing tables, firewall rules, and ports. Usually paired with a virtual Ethernet pair (`veth`), linking the container's network stack to a host bridge (`docker0`).
- **MNT (Mount Points)**: Provides an isolated filesystem mount table. Mounts and unmounts inside the container do not affect the host.
- **IPC (Inter-Process Communication)**: Isolates System V IPC and POSIX message queues.
- **UTS (Hostnames)**: Allows the container to define its own hostname and domain name without modifying the host system.
- **USER (UID and GID Mappings)**: Maps host user and group IDs to different IDs inside the namespace (e.g., UID 1000 on the host can act as UID 0/root inside the container for rootless execution).
- **CGROUP (Cgroup Root)**: Masks the host's actual cgroup directory tree structure from the process.

#### 2\. Control Groups / cgroups (What the Process Can Use)

While namespaces restrict **visibility**, cgroups restrict and account for **hardware resource usage**.

- **Resource Limiting**: Caps how much memory, CPU quota, block I/O bandwidth, and network priority a group of processes can consume.
- **Accounting & Metrics**: Tracks usage (e.g., reporting memory pressure and page cache consumption).
- **Process Number Limits (`pids` controller)**: Prevents fork bombs by capping the total number of processes inside the container.
- **OOM Handling**: When a container exceeds its memory hard limit, the kernel's Out-Of-Memory (OOM) killer selectively terminates processes within that cgroup rather than bringing down host processes.

Linux environments predominantly use **cgroups v2**, which features a unified hierarchy where a single process tree controls all resource controllers simultaneously (unlike cgroups v1, which had separate trees for memory, cpu, and blkio).

#### 3\. Root Filesystems and Layering: OverlayFS & pivot\_root

Containers do not boot a kernel; they reuse the host kernel with an independent directory tree:

- **Layering via OverlayFS**: Docker images consist of immutable read-only layers. OverlayFS stacks these layers (`lowerdir`) with an ephemeral read-write upper layer (`upperdir`) and presents them as a single unified view (`merged`). Changes made inside the container write only to the `upperdir` via copy-on-write (CoW).
- **`pivot_root` vs `chroot`**: While `chroot` changes the apparent root directory for the current process, it has well-known escape paths (especially for processes with root privileges). `runc` mounts the container's root filesystem and uses `pivot_root(new_root, put_old)`. This system call moves the current root mount to a temporary directory inside the new root, swaps in the new root, and then unmounts the old host root—making escape from the mount namespace impossible via directory traversal.

#### 4\. Security Isolation: Capabilities & Seccomp

A root user (`UID 0`) inside a traditional container could theoretically control hardware or manipulate kernel modules if unrestricted. Two mechanisms strip dangerous abilities:

- **Linux Capabilities**: Breaks traditional `root` authority into granular privileges (e.g., `CAP_NET_BIND_SERVICE`, `CAP_SYS_ADMIN`, `CAP_CHOWN`). `runc` drops most dangerous capabilities by default, such as `CAP_SYS_ADMIN` (mounting filesystems, loading BPF programs) and `CAP_SYS_MODULE` (loading kernel modules).
- **Seccomp (Secure Computing Mode)**: Acts as a syscall firewall. The default Docker seccomp profile filters roughly 44 out of over 400 system calls (blocking syscalls like `reboot`, `kexec_load`, `mount`, and `sys_chroot`).

### Step-by-Step Execution Inside `runc`

```
+-------------------------------------------------------------+
| 1. Read OCI bundle config.json & prepare rootfs             |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| 2. clone() child process with CLONE_NEW* namespace flags    |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| 3. Attach child to cgroup paths in /sys/fs/cgroup           |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| 4. pivot_root() into the new rootfs & unmount old host root |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| 5. Drop Linux capabilities & load Seccomp BPF filter        |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| 6. execve() replaces runc child with the target app binary  |
+-------------------------------------------------------------+
```

Once `execve()` executes, the child process is running your container application in-place under the configured boundaries, while the parent `runc` process exits.
