---
title: "Linux boot process"
date: "2026-10-31"
---

The Linux boot process follows a deterministic sequence transitioning hardware execution from firmware, through low-level CPU modes and memory staging, up to the first user-space process.

1

Firmware Initialization (BIOS / UEFI)

**1.Firmware Initialization (BIOS / UEFI):**
- **Power-On Self-Test (POST):** The processor begins executing code at the reset vector (`0xFFFFFFF0` on x86/x64). Firmware tests hardware peripherals, discovers system RAM, and initializes basic bus topology.
- **Handover:**
	- **Legacy BIOS:** Searches configured boot disks for an active partition containing a 512-byte Master Boot Record (MBR) at sector 0 and loads it into memory at `0x7C00`.
	- **Modern UEFI:** Reads the GUID Partition Table (GPT), accesses the FAT-formatted EFI System Partition (ESP), and executes a pre-configured architecture-specific EFI binary (e.g., `/EFI/BOOT/BOOTX64.EFI` or `grubx64.efi`).

2

Bootloader Execution (GRUB 2)

**2.Bootloader Execution (GRUB 2):**
- **Stage 1 & Stage 1.5:** Under BIOS, Stage 1 (MBR) loads Stage 1.5 / `core.img` embedded in the unpartitioned space between MBR and the first sector. Under UEFI, the firmware directly executes `grubx64.efi`.
- **Stage 2:** Loads filesystem drivers (ext4, btrfs, xfs), parses `/boot/grub/grub.cfg`, displays the boot menu, and handles kernel command-line parameters.
- **Kernel & Initramfs Placement:** Loads the compressed kernel image (`vmlinuz`) and initial RAM filesystem (`initramfs` or `initrd`) into physical RAM, transitions the CPU from real mode / protected mode to 64-bit long mode, and jumps to the kernel entry point.

3

Kernel Startup & Hardware Probing

**3.Kernel Startup & Hardware Probing:**
- **Decompression & Memory Setup:** The kernel self-extracts, maps memory pages, sets up page tables, enables paging, and initializes core data structures (IDT, GDT, scheduler).
- **Device Discovery:** Identifies CPU topology, NUMA nodes, interrupts (APIC), and probes PCI/PCIe buses.
- **Driver & Subsystem Bringup:** Initializes core subsystems including memory management, timer ticks, virtual filesystem (VFS), and IPC.

4

Initramfs Staging (Early Userspace)

**4.Initramfs Staging (Early Userspace):**
- **Temporary Root:** The kernel mounts `initramfs` as a temporary root filesystem (`tmpfs`) in memory and executes `/init` (typically a minimal shell or systemd instance).
- **Storage Modules:** Loads specialized storage drivers not compiled statically into the kernel—such as device mappers for LUKS disk encryption, software RAID (`mdadm`), LVM volumes, or network root mounts (iSCSI, NFS).
- **Switch Root:** Mounts the real root filesystem (`/`) from persistent storage (read-only initially), moves mount points, and runs `switch_root` / `pivot_root` to wipe the temporary RAM filesystem and hand control to the permanent filesystem.

5

Initialization System (systemd / PID 1)

**5.Initialization System (systemd / PID 1):**
- **First Userspace Process:** The kernel executes `/sbin/init` (symlinked to `/usr/lib/systemd/systemd` on modern distributions), assigning it **PID 1**.
- **Target Traversal:** Reads dependency graphs defined in unit files, resolving boot targets (e.g., `sysinit.target`  $\to$ `basic.target`  $\to$ `multi-user.target` / `graphical.target`).
- **Mounts & Daemons:** Re-mounts the root filesystem read-write, mounts `/etc/fstab` entries, initializes device nodes via `systemd-udevd`, sets hostname and networking, and spawns background system daemons in parallel.
- **Display & Login:** Launches the display manager (`gdm`, `sddm`) or virtual terminal login prompts (`agetty`), completing the boot pipeline.
