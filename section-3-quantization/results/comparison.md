# Quantization Benchmark Comparison

## Model

Qwen/Qwen2.5-1.5B-Instruct

## Hardware

- GPU: Tesla T4
- CUDA: 12.8
- PyTorch: 2.11.0+cu128

---

## Benchmark Results

| Metric | BF16 | 4-bit NF4 |
|--------|------|-----------|
| Average Latency (s) | 6.1505 | 9.2313 |
| Throughput (tokens/sec) | 25.062 | 16.319 |
| RAM (GB) | 1.599 | 1.615 |
| Allocated VRAM (GB) | 2.884 | 1.084 |
| Peak VRAM (GB) | 2.887 | 1.113 |

---

## Analysis

The NF4 quantized model reduced GPU memory consumption significantly,
decreasing allocated VRAM from approximately
2.884 GB
to
1.084 GB.

However, for this benchmark on a Tesla T4 GPU using
Qwen2.5-1.5B-Instruct,
the BF16 model achieved lower latency and higher throughput.

This demonstrates the common trade-off of quantization:

- Lower memory footprint
- Ability to run on smaller GPUs
- Possible reduction in inference speed depending on hardware and model size

For this workload, BF16 provides the best inference performance,
while NF4 offers substantially better memory efficiency.
