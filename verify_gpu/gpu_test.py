#!/usr/bin/env python3
import numpy as np
import time
import os

print("Testing GPU availability...")

# Check if CUDA is available through environment variables
print("\nCUDA Environment Variables:")
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

try:
    import cupy as cp
    print("\nCuPy Information:")
    print(f"CuPy version: {cp.__version__}")
    
    # Check CUDA version
    print(f"CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}")
    
    # Get device information
    num_devices = cp.cuda.runtime.getDeviceCount()
    print(f"Number of CUDA devices: {num_devices}")
    
    for device_id in range(num_devices):
        device_props = cp.cuda.runtime.getDeviceProperties(device_id)
        print(f"\nDevice {device_id}: {device_props['name'].decode()}")
        print(f"  Compute capability: {device_props['major']}.{device_props['minor']}")
        print(f"  Total memory: {device_props['totalGlobalMem'] / (1024**3):.2f} GB")
    
    # Simple performance test
    print("\nPerformance Test:")
    
    # Create large arrays on CPU
    size = 4000
    print(f"Creating {size}x{size} arrays...")
    
    # CPU matrix multiplication
    a_cpu = np.random.random((size, size)).astype(np.float32)
    b_cpu = np.random.random((size, size)).astype(np.float32)
    
    print("Starting CPU matrix multiplication...")
    start_time = time.time()
    c_cpu = np.dot(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.2f} seconds")
    
    # GPU matrix multiplication
    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)
    
    print("Starting GPU matrix multiplication...")
    start_time = time.time()
    c_gpu = cp.dot(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()  # Wait for completion
    gpu_time = time.time() - start_time
    print(f"GPU time: {gpu_time:.2f} seconds")
    
    # Verify results match
    c_gpu_cpu = cp.asnumpy(c_gpu)
    max_diff = np.max(np.abs(c_cpu - c_gpu_cpu))
    print(f"Maximum difference between CPU and GPU results: {max_diff}")
    
    # Calculate speedup
    speedup = cpu_time / gpu_time
    print(f"\nGPU is {speedup:.2f}x faster than CPU")
    
    print("\nGPU test successful! CuPy is properly using the GPU.")
    
except ImportError:
    print("\nCuPy is not installed. Please install it with:")
    print("pip install cupy-cuda11x")

except Exception as e:
    print(f"\nError: {str(e)}")
    print("\nGPU test failed. Please check your CUDA installation and cupy version.") 