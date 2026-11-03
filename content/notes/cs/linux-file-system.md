---
title: "Linux file system"
date: "2026-10-31"
---

In Linux, everything is treated as a file—including directories, hardware devices, and network sockets—organized under a single, unified hierarchical tree starting at the root directory (`/`).

Unlike Windows, which uses separate drive letters (`C:\`, `D:\`), Linux mounts all physical drives, partitions, and external media into specific branches of this single directory tree according to the **Filesystem Hierarchy Standard (FHS)**.

![Linux Directory Hierarchy Structure, AI generated](https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcS1yRkQvLEtxrW3aPY_N8CfOJS5rGToBt4siX5kHsmsy872CTiSYdjfaSsoVCi4DbE3rVwYH5VUJ259xpE)

Linux Directory Hierarchy Structure. Source: Made with Google AI

### Core Directory Breakdown

| Directory | Purpose | Typical Contents |
| --- | --- | --- |
| **`/`** | Root directory | Top level of the filesystem; everything starts here. |
| **`/bin`** | Essential user binaries | Core commands needed for single-user mode and booting (`ls`, `cp`, `bash`, `cat`). |
| **`/sbin`** | System binaries | Administrative commands reserved for root/system recovery (`iptables`, `fdisk`, `reboot`). |
| **`/etc`** | System configuration | Machine-local host configuration files and startup scripts (`fstab`, `passwd`, `nginx/`). |
| **`/home`** | User personal files | Stores personal directories (`/home/username`) for documents, downloads, and user settings. |
| **`/root`** | Root user home | The dedicated home directory for the superuser (root). |
| **`/var`** | Variable data | Files that grow dynamically during runtime (logs in `/var/log`, mail spools, databases). |
| **`/tmp`** | Temporary files | Volatile scratch space; usually cleared automatically on reboot. |
| **`/usr`** | User system resources | Read-only user data, second-tier binaries (`/usr/bin`), libraries (`/usr/lib`), and docs. |
| **`/opt`** | Optional software | Third-party or self-contained application packages (e.g., Google Chrome, proprietary tools). |
| **`/dev`** | Device nodes | Special files representing hardware devices (disks like `sda`, terminals like `tty`, `/dev/null`). |
| **`/proc`** & **`/sys`** | Virtual filesystems | Dynamic runtime metadata exposed by the kernel; exists in RAM rather than disk. |
| **`/mnt`** & **`/media`** | Mount points | Temporary mount points (`/mnt` for manual mounts, `/media` for auto-mounted USB/removable drives). |

### Key Concepts

- **Everything is a File:** Linux interfaces with hardware and processes using file descriptors. Reading from `/dev/urandom` produces random bytes; writing to `/dev/null` discards data.
- **Inodes (Index Nodes):** Disk storage separates file metadata from file content. An inode stores file type, permissions, size, timestamps, and data block locations. The directory entry simply maps a human-readable filename to an inode number.
- **Permissions & Ownership:** Every file belongs to an owner and a group, carrying read (`r`), write (`w`), and execute (`x`) permissions evaluated across three tiers: User, Group, and Others.
- **Underlying Filesystems:** While the directory tree provides a standard logical view, Linux formats storage using specialized filesystems such as **ext4** (default standard), **XFS** (high-throughput, large data), and **Btrfs** or **ZFS** (snapshots, volume management).
