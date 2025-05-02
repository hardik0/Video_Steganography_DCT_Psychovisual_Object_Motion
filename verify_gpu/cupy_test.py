#!/usr/bin/env python3
import numpy as np
import time
import sys

print(f"Python version: {sys.version}")
print("Testing cupy...")

try:
    import cupy as cp
    print(f"cupy version: {cp.__version__}")
    
    # Check CUDA version
    print(f"CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}")
    
    # Get device info
    num_devices = cp.cuda.runtime.getDeviceCount()
    print(f"Number of CUDA devices: {num_devices}")
    
    if num_devices > 0:
        device_props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"Device name: {device_props['name'].decode()}")
        print(f"Total memory: {device_props['totalGlobalMem'] / (1024**3):.2f} GB")
        
        # Create and transfer data
        print("\nCreating arrays...")
        a_np = np.arange(10000000, dtype=np.float32)
        b_np = np.arange(10000000, dtype=np.float32)
        
        # CPU computation
        start = time.time()
        c_np = a_np + b_np
        cpu_time = time.time() - start
        print(f"CPU time: {cpu_time:.6f} seconds")
        
        # GPU computation
        start = time.time()
        a_cp = cp.array(a_np)
        b_cp = cp.array(b_np)
        transfer_time = time.time() - start
        print(f"Transfer to GPU time: {transfer_time:.6f} seconds")
        
        start = time.time()
        c_cp = a_cp + b_cp
        cp.cuda.Stream.null.synchronize()
        gpu_compute_time = time.time() - start
        print(f"GPU computation time: {gpu_compute_time:.6f} seconds")
        
        start = time.time()
        c_cp_np = cp.asnumpy(c_cp)
        transfer_back_time = time.time() - start
        print(f"Transfer back to CPU time: {transfer_back_time:.6f} seconds")
        
        # Check results
        print(f"Results match: {np.allclose(c_np, c_cp_np)}")
        
        # Total GPU time including transfers
        total_gpu_time = transfer_time + gpu_compute_time + transfer_back_time
        print(f"\nTotal CPU time: {cpu_time:.6f} seconds")
        print(f"Total GPU time (including transfers): {total_gpu_time:.6f} seconds")
        print(f"Speedup (including transfers): {cpu_time / total_gpu_time:.2f}x")
        print(f"Speedup (computation only): {cpu_time / gpu_compute_time:.2f}x")
        
        print("\nCuPy test passed successfully!")
    else:
        print("No CUDA devices found.")
except ImportError:
    print("cupy is not installed. Please install it with: pip install cupy-cuda11x")
except Exception as e:
    print(f"Error testing cupy: {str(e)}") 