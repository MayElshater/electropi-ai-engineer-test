# Section 3 – Model Quantization Benchmark

## Objective

Evaluate the impact of 4-bit NF4 quantization on memory usage and inference performance using Qwen2.5-1.5B-Instruct.

## Model

- Qwen/Qwen2.5-1.5B-Instruct

## Environment

- Google Colab
- NVIDIA Tesla T4
- PyTorch
- Transformers
- bitsandbytes

## Configurations

- BF16
- 4-bit NF4

## Metrics

- Latency
- Throughput (tokens/sec)
- RAM Usage
- GPU Memory (VRAM)

## Results

The NF4 model reduced GPU memory consumption significantly while introducing higher latency and lower throughput on Tesla T4. This demonstrates the trade-off between memory efficiency and inference speed.

## Half-Page Technical Write-up

For this benchmark, I used Qwen2.5-1.5B-Instruct and compared a BF16 model with a 4-bit NF4 quantized model using BitsAndBytes on a Tesla T4 GPU. The experiment showed that NF4 significantly reduced GPU memory usage while increasing latency and reducing throughput, illustrating the common trade-off between memory efficiency and inference performance.

From this experience, I would choose BitsAndBytes during experimentation and model development. It integrates directly with the Hugging Face Transformers ecosystem, requires minimal changes to existing code, and makes it easy to switch between full precision and quantized inference. This flexibility is particularly useful for benchmarking different models or validating quality before deployment.

For a production GPU deployment, I would prefer GPTQ or AWQ over BitsAndBytes when inference latency is more important than flexibility. Both GPTQ and AWQ produce static quantized model weights that are optimized ahead of time, eliminating the runtime quantization overhead introduced by BitsAndBytes. AWQ is especially attractive because it preserves important activation-aware weights, often providing better quality at low-bit precision while maintaining high inference speed.

When deploying on CPU-only systems or edge devices, I would choose GGUF instead of either GPTQ or BitsAndBytes. GGUF is specifically designed for inference with llama.cpp and provides excellent CPU performance, efficient memory usage, and a simple deployment workflow. In Section 4, I used a GGUF version of the same model with llama-cpp-python to build a lightweight inference API, making it a practical choice for local deployment without requiring CUDA or specialized GPU libraries.

Overall, my selection depends primarily on the deployment target: BitsAndBytes for experimentation, GPTQ or AWQ for optimized GPU inference, and GGUF for lightweight CPU-based production services.