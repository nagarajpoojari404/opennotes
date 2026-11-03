---
title: "Projects"
summary: "Open-source distributed systems, compilers, database engines, machine learning frameworks, and developer tools built from scratch."
cascade:
  type: docs
sidebar:
  open: true
---

A collection of open-source projects exploring distributed systems, compiler design, storage engines, machine learning infrastructure, computer graphics, and developer tooling. Built from fundamental principles to understand how complex systems operate under the hood.

---

{{< cards cols="2" >}}
  {{< card link="picasso/" title="Picasso Language" subtitle="Modern compiled systems language with lightweight green threads, automatic memory management, and zero-cost abstractions." tag="Compiler / C" >}}
  {{< card link="orange/" title="OrangeDB" subtitle="Distributed, cloud-native NoSQL key-document database with LSM storage engine (Parrot), K8s operator, and quorum consensus." tag="Database / Go" >}}
  {{< card link="ripple/" title="Ripple MQ" subtitle="High-performance, single-threaded non-blocking in-memory event streaming broker written in Go for massive throughput." tag="Streaming / Go" >}}
  {{< card link="panda-fs/" title="PandaFS (nanoDFS)" subtitle="Fault-tolerant, GFS-inspired distributed file system featuring Master-Slave chunk orchestration and client-side encryption." tag="Storage / Go" >}}
  {{< card link="mint/" title="Mint" subtitle="Compute-optimal distributed LLM pre-training & fine-tuning framework (SFT/DPO) optimized for Apple Silicon (Metal) and NVIDIA GPUs." tag="AI / PyTorch" >}}
  {{< card link="owl/" title="Owl Graphics Engine" subtitle="High-level 3D graphics and real-time ray tracing engine with rigid-body physics simulation built from scratch in OpenGL/C++." tag="Graphics / C++" >}}
  {{< card link="x-linux/" title="x-Linux" subtitle="Custom bootable Linux distribution and automated build pipeline created following Linux From Scratch specifications." tag="OS / Shell" >}}
  {{< card link="synapc/" title="SynapC" subtitle="Minimalist neural network engine written in pure C from scratch with Python bindings and zero external dependencies." tag="Deep Learning / C" >}}
  {{< card link="quantum-threads/" title="Quantum Threads" subtitle="Hybrid Quantum-Classical machine learning using Parameterized Quantum Circuits (PQC) and distributed cloud QPUs." tag="Quantum / Qiskit" >}}
  {{< card link="video-interpolation-ai/" title="Video Interpolation AI" subtitle="Vision Transformer (VIFT) with Swin attention blocks for high-fidelity video frame interpolation and FPS upscaling." tag="Vision / PyTorch" >}}
  {{< card link="gluon-ai/" title="Gluon AI" subtitle="Intelligent VS Code extension for automated code explanation and documentation generation using LLMs." tag="DevTool / TypeScript" >}}
  {{< card link="devpage/" title="DevPage" subtitle="Developer social platform and portfolio showcase ('Instagram for developers') with AWS Cognito and GitHub authentication." tag="Full Stack / Node.js" >}}
  {{< card link="chess/" title="Chess Platform" subtitle="Real-time multiplayer chess platform with WebRTC live video/audio streaming, Socket.io synchronization, and AI coach." tag="Full Stack / Socket.io" >}}
  {{< card link="ascii-art/" title="ASCII Art" subtitle="Client-side real-time video stream to ASCII character art converter built with p5.js and canvas manipulation." tag="Creative / JavaScript" >}}
{{< /cards >}}

---

## All Projects Summary

| Project | Domain | Tech Stack | Highlights | Links |
| :--- | :--- | :--- | :--- | :--- |
| **[Picasso](picasso/)** | Compilers / Languages | C, LLVM, Assembly | Native compilation, green threads, GC, static typing | [GitHub](https://github.com/nagarajpoojari404/picasso-lang) · [Dedicated Page](picasso/) |
| **[OrangeDB](orange/)** | Distributed Databases | Go, Raft, LSM-Tree, K8s | Parrot LSM engine, JSON documents, Orangectl K8s operator | [GitHub](https://github.com/nagarajpoojari404/orange) · [Dedicated Page](orange/) |
| **[Ripple MQ](ripple/)** | Event Streaming | Go, TCP, Custom Binary Protocol | Single-threaded non-blocking I/O, low latency, partition streams | [GitHub](https://github.com/ripple-mq/ripple-server) · [Dedicated Page](ripple/) |
| **[PandaFS](panda-fs/)** | Distributed File Systems | Go, gRPC, P2P, Cryptography | Master/Slave chunk servers, client-side encryption, token ACL | [GitHub](https://github.com/nanoDFS) · [Dedicated Page](panda-fs/) |
| **[Mint](mint/)** | Distributed LLM Training | Python, PyTorch, Metal, CUDA | Compute-optimal scaling, ZeRO-2, MuonAdamW, SFT & DPO | [GitHub](https://github.com/nagarajpoojari404/mint) · [Dedicated Page](mint/) |
| **[Owl Engine](owl/)** | 3D Graphics / Physics | C++, OpenGL, GLSL | Real-time ray tracing, physics engine (gravity, collision) | [GitHub](https://github.com/nagarajpoojari404/owl) · [Dedicated Page](owl/) |
| **[x-Linux](x-linux/)** | Operating Systems | Shell, C, Linux Kernel | Custom Linux distro build pipeline, bootable USB ISO | [GitHub](https://github.com/nagarajpoojari404/x-Linux) · [Dedicated Page](x-linux/) |
| **[SynapC](synapc/)** | Neural Networks | C, Python C-API | Pure C neural network from scratch, PyPI package | [GitHub](https://github.com/nagarajpoojari404/SynapC) · [Dedicated Page](synapc/) |
| **[Quantum Threads](quantum-threads/)** | Quantum ML | Python, Qiskit, PyTorch | Hybrid PQC + ResNet/Inception, adjoint differentiation | [GitHub](https://github.com/nagarajpoojari404/Quantum-Threads) · [Dedicated Page](quantum-threads/) |
| **[Video Interpolation AI](video-interpolation-ai/)** | Computer Vision | Python, PyTorch, Transformers | Swin Transformer (VIFT), frame rate upscaling, PSNR/SSIM gains | [GitHub](https://github.com/nagarajpoojari404/video-interpolation-AI) · [Dedicated Page](video-interpolation-ai/) |
| **[Gluon AI](gluon-ai/)** | Developer Tooling | TypeScript, VS Code API, OpenAI | Automated code documentation generator, VS Code extension | [GitHub](https://github.com/nagarajpoojari404/Gluon-AI) · [Dedicated Page](gluon-ai/) |
| **[DevPage](devpage/)** | Web Platforms | Node.js, Express, MongoDB, AWS | Social media for developers, AWS Cognito, GitHub OAuth | [GitHub](https://github.com/nagarajpoojari404/DevPage) · [Dedicated Page](devpage/) |
| **[Chess Platform](chess/)** | Real-Time Web | Node.js, Socket.io, PeerJS, Python | Live WebRTC video/audio chess matches, AI chess coach | [GitHub](https://github.com/nagarajpoojari404/chessProject) · [Dedicated Page](chess/) |
| **[ASCII Art](ascii-art/)** | Creative Coding | JavaScript, p5.js, HTML5 Canvas | Real-time camera feed to ASCII renderer | [GitHub](https://github.com/nagarajpoojari404/ASCII-Art) · [Dedicated Page](ascii-art/) |
