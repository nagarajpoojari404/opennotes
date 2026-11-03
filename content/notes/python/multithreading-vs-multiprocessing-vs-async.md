---
title: "Multithreading vs multiprocessing vs async"
date: "2026-10-27"
---

In Python, concurrent programming revolves around three primary paradigms: **Multithreading**, **Multiprocessing**, and **Asynchronous Programming (`asyncio`)**.

Each approach targets a specific type of bottleneck—either **CPU-bound** (computationally expensive tasks) or **I/O-bound** (waiting for external events like network, disk, or databases).

## 1\. Overview & Comparison

| Feature | Multithreading (`threading`) | Multiprocessing (`multiprocessing`) | Asynchronous (`asyncio`) |
| --- | --- | --- | --- |
| **Model** | Preemptive multitasking | Preemptive multitasking | Cooperative multitasking |
| **Concurrency Type** | Shared memory, multiple threads | Separate processes, isolated memory | Single thread, event loop |
| **Impact of GIL** | **Constrained** (1 thread active at a time for Python bytecode) | **Bypassed** (1 GIL per OS process) | **N/A** (runs in a single thread) |
| **Ideal Use Case** | I/O-bound tasks (disk, basic web scraping) | CPU-bound tasks (data processing, math, image crunching) | High-concurrency I/O-bound tasks (web servers, websockets) |
| **Memory Overhead** | Low (threads share process memory) | High (each process has its own memory space) | Very Low (single thread, light coroutines) |
| **Context Switching** | Managed by OS | Managed by OS | Managed by application (via `await`) |

## 2\. In-Depth Breakdown & Internal Mechanics

### Paradigm 1: Multithreading (`threading`)

#### How it works internally

A thread is the smallest unit of execution that an Operating System can schedule. Python's `threading` module uses native OS threads (POSIX threads on Linux/macOS, Windows threads on Windows).

However, CPython (the standard Python interpreter) features the **Global Interpreter Lock (GIL)**—a mutual-exclusion lock that prevents multiple OS threads from executing Python bytecode simultaneously.

```
[ OS Thread 1 ]  --->  ( Holds GIL )  ---> Executing Python Bytecode
                                     |
       [ OS Thread 2 ]  --->  ( Blocked )   ---> Waiting for GIL
```
- **GIL Release during I/O:** When a thread performs an I/O operation (reading a socket, writing to a file, calling `time.sleep()`), CPython explicitly releases the GIL. Another thread can then acquire it and execute code while the first thread waits on the OS kernel.
- **Preemption:** For pure Python code, CPython periodically forces the running thread to release the GIL (based on a interval timer, defaulting to 5ms) to allow other threads a chance to run.

#### Code Example: Downloading URLs (I/O-Bound)

Python

```python
import concurrent.futures
import time

def fetch_data(task_id: int):
    print(f"[Thread {task_id}] Starting fetch...")
    time.sleep(1.5)  # Simulates network I/O; releases GIL
    print(f"[Thread {task_id}] Done.")
    return f"Data from task {task_id}"

def run_threads():
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch_data, range(3)))
    
    print(f"Threads completed in {time.perf_counter() - start:.2f} seconds")

if __name__ == "__main__":
    run_threads()
```

### Paradigm 2: Multiprocessing (`multiprocessing`)

#### How it works internally

Instead of spawning threads inside a single Python process, `multiprocessing` spawns entirely separate OS processes—each with its own independent CPython interpreter, memory space, and GIL.

```
[ Process 1 (Master) ] --->  GIL 1  --->  CPU Core 0
 [ Process 2 (Worker) ] --->  GIL 2  --->  CPU Core 1
 [ Process 3 (Worker) ] --->  GIL 3  --->  CPU Core 2
```

Because each process has its own GIL, Python code can execute across multiple CPU cores in parallel, fully bypassing the GIL restriction.

- **Process Creation Mechanics:**
	- **`fork` (Unix standard):** Clones the parent process's memory space using Copy-on-Write (COW).
		- **`spawn` (Windows & default on macOS):** Starts a fresh Python interpreter process, importing the module from scratch.
- **Inter-Process Communication (IPC):** Because memory is isolated, processes cannot directly read/write shared variables. Communication requires IPC mechanisms like `multiprocessing.Queue`, `Pipe`, or `SharedMemory`, which incur serialization (`pickle`) overhead.

#### Code Example: Heavy Computation (CPU-Bound)

Python

```python
import concurrent.futures
import time

def compute_heavy_task(n: int):
    # Pure CPU calculation
    return sum(i * i for i in range(n))

def run_multiprocessing():
    numbers = [10_000_000, 10_000_000, 10_000_000]
    
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(compute_heavy_task, numbers))
    
    print(f"Multiprocessing completed in {time.perf_counter() - start:.2f} seconds")

if __name__ == "__main__":
    run_multiprocessing()
```

### Paradigm 3: Asynchronous Programming (`asyncio`)

#### How it works internally

`asyncio` uses **single-threaded cooperative multitasking**. Rather than letting the OS switch contexts unpredictably, Python explicitly yields control when waiting for I/O using an **Event Loop**.

```
+------------------------------------------+
              |               Event Loop                 |
              +------------------------------------------+
                /             |              \
      Coroutine A            Coroutine B       Coroutine C
   (Awaiting Network)      (Active Executing)  (Awaiting Timer)
```
- **Event Loop Engine:** Under the hood, `asyncio` registers network sockets and descriptors with high-performance OS I/O multiplexing primitives (`epoll` on Linux, `kqueue` on macOS, `IOCP` on Windows).
- **Coroutines & `await`:** Functions defined with `async def` produce coroutines. Calling `await` yields control back to the event loop: *"I am blocked on I/O; run other tasks until my result is ready."*
- **Zero Thread Context-Switching:** Because everything runs on a single thread, there are no OS context switches, thread stack memory overheads, or race conditions around shared memory (as long as synchronization points at `await` are managed).

#### Code Example: Async Network Requests (High Concurrency I/O)

Python

```python
import asyncio
import time

async def async_fetch(task_id: int):
    print(f"[Async {task_id}] Starting...")
    # Yields control back to event loop without blocking thread
    await asyncio.sleep(1.5) 
    print(f"[Async {task_id}] Done.")
    return f"Result {task_id}"

async def main():
    start = time.perf_counter()
    
    # Schedule all 3 coroutines concurrently on the event loop
    tasks = [async_fetch(i) for i in range(3)]
    results = await asyncio.gather(*tasks)
    
    print(f"Async completed in {time.perf_counter() - start:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
```
