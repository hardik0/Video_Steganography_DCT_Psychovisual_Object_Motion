# GPU Acceleration for DCT-based Video Steganography

This document discusses the implementation and benefits of GPU acceleration in our video steganography system.

## Implementation Details

We have implemented GPU acceleration using the following components:

1. **dct_utils_gpu.py**: Core GPU-accelerated utility functions using CUDA/cupy
2. **embed_gpu.py**: GPU-optimized embedding implementation
3. **extract_gpu.py**: GPU-optimized extraction implementation

The GPU acceleration focuses on the most computationally intensive operations:
- **DCT/IDCT calculations**: Parallel processing of DCT coefficients for blocks
- **Batch processing**: Processing multiple blocks simultaneously
- **Motion detection**: Using CUDA for motion analysis between frames

## Required Dependencies

To use GPU acceleration, the following dependencies are required:
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- cupy library installed with appropriate CUDA version (e.g., `cupy-cuda11x`)

## Fallback Mechanism

The implementation includes robust fallback mechanisms:
- Automatically detects if CUDA/cupy is available
- Falls back to CPU processing when GPU is not available
- Provides detailed diagnostics about GPU availability

## Current Performance Results

Based on our testing on the current system with an NVIDIA GeForce GTX 1660 Ti:

```
# GPU vs CPU Performance Comparison

## Embedding Performance

| Metric                 |   CPU |   GPU | Speedup   |
|:-----------------------|------:|------:|:----------|
| Raw Execution Time (s) | 22.33 | 25.80 | 0.87x     |
| Average PSNR (dB)      | 59.61 | 59.61 | +0.00     |
| Output File Size (MB)  |  5.11 |  5.11 | 1.00x     |

## Extraction Performance

| Metric                 |   CPU |   GPU | Speedup   |
|:-----------------------|------:|------:|:----------|
| Raw Execution Time (s) |  1.78 |  2.53 | 0.70x     |
| Bit Error Rate (%)     |  2.04 |  2.04 | +0.00     |
```

**Performance Analysis:**

Despite having CuPy properly installed and the GPU being detected (verified through `gpu_test.py`), we're still observing that the GPU implementation is slower than the CPU implementation. This is likely due to several factors:

1. **Data Transfer Overhead**: There's significant overhead in transferring data between CPU and GPU memory, which is especially noticeable for smaller workloads.

2. **Small Computation Batches**: The DCT operations are performed on small 8x8 blocks, which don't fully utilize the GPU's parallel processing capabilities.

3. **OpenCV Limitations**: OpenCV was not built with CUDA support on this system, as confirmed by `opencv_check.py`, limiting acceleration to only CuPy operations.

4. **Motion Detection**: The motion detection component, which takes up to 55.8% of the processing time in embedding, still runs primarily on the CPU.

For the current workload size and video resolution, the added overhead of GPU operations outweighs the potential speedup from parallel processing.

## Expected Performance Improvements

GPU acceleration typically provides better results with:
1. **Larger video resolutions**: Higher resolution videos (4K+) would provide more data for parallel processing
2. **Longer videos with many frames**: More frames means more blocks to process in parallel
3. **Batch processing optimization**: Processing multiple frames simultaneously would better utilize the GPU

## Future Enhancements

Planned enhancements for GPU acceleration:
1. **Batch processing optimization**: Process multiple frames simultaneously
2. **Memory management improvements**: Reduce CPU-GPU transfers
3. **Support for AMD GPUs via ROCm framework**
4. **Full motion detection on GPU**: Move more of the motion detection algorithm to run on the GPU
5. **OpenCV with CUDA**: Build or use a version of OpenCV with CUDA support

## Conclusion

While GPU acceleration is currently implemented and functional, the specific workload characteristics of our video steganography system don't fully benefit from GPU parallelization at typical video resolutions. For larger videos or batch processing scenarios, the GPU implementation may still provide benefits.

The implementation gracefully falls back to CPU processing when GPU resources are unavailable, ensuring the code works across different environments.

To enable GPU acceleration, install the required dependencies:
```
pip install cupy-cuda11x  # Use appropriate version for your CUDA installation
``` 