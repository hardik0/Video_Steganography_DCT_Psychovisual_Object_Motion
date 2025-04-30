#!/usr/bin/env python3
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")

# Check CUDA availability in OpenCV
try:
    count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA enabled devices in OpenCV: {count}")
    if count > 0:
        print("OpenCV was built with CUDA support and can access your GPU")
        
        # Print device information
        for i in range(count):
            device = cv2.cuda.getDevice()
            print(f"Current CUDA device: {device}")
            
            # Print device properties
            props = cv2.cuda.DeviceInfo()
            print(f"Device name: {props.name()}")
            print(f"Compute capability: {props.majorVersion()}.{props.minorVersion()}")
            print(f"Multi-processor count: {props.multiProcessorCount()}")
            print(f"Clock rate: {props.clockRate()} kHz")
            print(f"Total memory: {props.totalMemory() / (1024**3):.2f} GB")
            print(f"Free memory: {props.freeMemory() / (1024**3):.2f} GB")
            
        # Test a simple CUDA operation in OpenCV
        print("\nTesting a simple CUDA operation...")
        img = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_GRAY2BGR)
        result = gpu_gray.download()
        
        print("CUDA operation completed successfully!")
    else:
        print("No CUDA-capable devices are detected by OpenCV")
except Exception as e:
    print(f"CUDA is not supported in this OpenCV build: {e}")
    print("You may need to rebuild OpenCV with CUDA support") 