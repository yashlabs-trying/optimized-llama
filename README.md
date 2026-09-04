# Optimized Llama

Optimizing Llama for GPU inference speed and throughput. Current target: **Llama-3.2-3B** on an **NVIDIA A40**.

## Goal

5-7x faster than naive PyTorch/HuggingFace fp16 inference, with **8-bit (INT8) quantization only**.

## Hardware Target

| Component | Spec |
|-----------|------|
| GPU | NVIDIA A40 (Ampere), 45 GB VRAM |
| Bandwidth | ~696 GB/s |
| CPU / RAM | 96 cores / 503 GB |
| Driver | 580.159.04 / CUDA 13.0 |

## Optimization Layers (in order of impact for throughput)

1. **INT8 SmoothQuant (W8A8)** - ~2x (halves weight bytes read per token)
2. **FlashAttention-2 + kernel fusion** - ~1.3-1.5x (less HBM traffic)
3. **PagedAttention + KV cache quantization** - enables larger batches
4. **Continuous batching + chunked prefill** - 3-10x (the biggest lever)
5. **CUDA graphs** - ~1.2-1.5x (reduces launch overhead on small models)
6. **Speculative decoding** - optional, latency-focused

## Project Structure

```
optimized-llama/
├── kernels/          # Custom CUDA kernels
├── quantization/     # Quantization utilities (SmoothQuant, INT8)
├── attention/        # Flash attention implementation
├── inference/        # Optimized inference engine
├── benchmarks/       # TTFT / ITL / throughput harness
├── profiling/        # Nsight Compute / Systems scripts
└── scripts/          # Environment + model setup
```

## Workflow

```
write code -> git commit -> push -> pull on GPU -> test -> profile -> stop GPU -> iterate
```

Models are downloaded **only on the GPU workspace** and are gitignored (never committed).

## Profiling

Three measurement sets:

| Set | Tool | Captures |
|-----|------|----------|
| Kernel-level | Nsight Compute (`ncu`) | per-kernel SM/DRAM utilization |
| System timeline | Nsight Systems (`nsys`) | CPU/GPU overlap, launch gaps |
| Serving metrics | benchmark harness | TTFT, ITL, throughput |

## Quick Start

```bash
bash scripts/setup_env.sh
bash scripts/download_model.sh   # requires HF token
python benchmarks/bench_serving.py
```

## License

MIT License
