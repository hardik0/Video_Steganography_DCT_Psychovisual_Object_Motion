# GPU Optimization Roadmap

This document outlines a detailed roadmap for optimizing our GPU-accelerated video steganography system.

## 1. Batch Processing Implementation

### Current Issue
Processing individual 8x8 DCT blocks incurs significant overhead for each GPU transfer.

### Solution: Frame Batching
Modify the code to process multiple frames at once:

```python
# Current implementation (one frame at a time)
for frame in frames:
    # Process single frame
    process_frame(frame)

# Improved implementation (batch of frames)
batch_size = 16  # Tune this based on GPU memory
for i in range(0, len(frames), batch_size):
    batch = frames[i:i+batch_size]
    process_frame_batch(batch)
```

### Solution: Block Batching
Improve the DCT batch processing:

```python
# Current batch implementation can be enhanced further
def parallel_dct_batch(blocks, use_gpu=False):
    # Current implementation loops through blocks on GPU
    for i in range(len(blocks)):
        result_gpu[i] = cuda_dct(cuda_dct(batch_gpu[i].T, norm='ortho').T, norm='ortho')
    
    # Improved version - reshape to process all blocks at once
    # This avoids Python loop overhead on GPU
    batch_shape = batch_gpu.shape
    # Reshape to 2D array where each row is a flattened block
    reshaped = batch_gpu.reshape(batch_shape[0], -1)
    # Process in a single operation if possible
    # (Implementation depends on the specific DCT function capabilities)
```

## 2. Memory Optimization

### Current Issue
Frequent data transfers between CPU and GPU memory.

### Solution: Keep Data on GPU
Minimize transfers by keeping data on the GPU:

```python
# Current approach
for frame in frames:
    # Transfer to GPU
    frame_gpu = cp.asarray(frame)
    # Process
    result_gpu = process_on_gpu(frame_gpu)
    # Transfer back to CPU
    result = cp.asnumpy(result_gpu)
    # Use result...

# Improved approach
# Transfer all frames at once (or in large batches)
frames_gpu = cp.asarray(frames)
results_gpu = []
for i in range(len(frames)):
    # Keep all intermediate results on GPU
    result_gpu = process_on_gpu(frames_gpu[i])
    results_gpu.append(result_gpu)
# Transfer final results back to CPU only at the end
results = cp.asnumpy(cp.stack(results_gpu))
```

### Solution: GPU Memory Management
Implement proper memory management:

```python
# Free memory when not needed
cp.get_default_memory_pool().free_all_blocks()

# For large videos, implement incremental processing
def process_large_video(video_path, output_path):
    # Process video in chunks to avoid GPU memory overflow
    chunk_size = 100  # frames
    # Open video
    cap = cv2.VideoCapture(video_path)
    # ...
    while True:
        frames = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        if not frames:
            break
            
        # Process chunk
        process_frames_on_gpu(frames)
        
        # Clear GPU memory after each chunk
        cp.get_default_memory_pool().free_all_blocks()
```

## 3. Algorithm Adaptations

### Current Issue
Motion detection is a major bottleneck and runs primarily on CPU.

### Solution: GPU-Accelerated Motion Detection

```python
def detect_motion_gpu(prev_frame, curr_frame, threshold=30):
    """GPU-accelerated motion detection between frames"""
    if prev_frame is None or curr_frame is None:
        return None
    
    # Convert frames to grayscale if needed
    if len(prev_frame.shape) > 2:
        # Use GPU for color conversion if OpenCV has CUDA
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
    
    if len(curr_frame.shape) > 2:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame
    
    # Transfer to GPU just once
    prev_gpu = cp.asarray(prev_gray)
    curr_gpu = cp.asarray(curr_gray)
    
    # Calculate absolute difference
    frame_diff_gpu = cp.abs(curr_gpu - prev_gpu)
    
    # Threshold operation on GPU
    motion_mask_gpu = cp.zeros_like(frame_diff_gpu)
    motion_mask_gpu[frame_diff_gpu > threshold] = 255
    
    # Apply morphological operations (if CuPy provides this functionality)
    # Otherwise, transfer back for this step
    
    return cp.asnumpy(motion_mask_gpu)
```

### Solution: DCT Coefficient Optimization
Optimize DCT calculations for GPU architecture:

```python
# Current approach processes blocks individually
# Consider using cuFFT for larger batch operations if applicable
```

## 4. OpenCV with CUDA Support

### Current Issue
Our OpenCV installation doesn't have CUDA support.

### Solution
Install OpenCV with CUDA support:

```bash
# Example steps for building OpenCV with CUDA
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D OPENCV_ENABLE_NONFREE=ON \
      ..
make -j$(nproc)
sudo make install
```

Then adapt our code to use OpenCV's CUDA modules:

```python
# Check if OpenCV CUDA is available
if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Use OpenCV CUDA functions
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    
    # Example: CUDA-accelerated blur
    gpu_result = cv2.cuda.blur(gpu_frame, (5, 5))
    result = gpu_result.download()
else:
    # Fall back to CPU
    result = cv2.blur(frame, (5, 5))
```

## 5. Implementation Strategy

1. **Benchmark Current Hotspots**
   - Profile the code to identify the most time-consuming operations
   - Focus optimization on these areas first

2. **Implement Incremental Changes**
   - Start with batch processing of frames
   - Then optimize memory transfers
   - Finally, adapt algorithms for GPU

3. **Continuous Testing**
   - Test with various video sizes and resolutions
   - Measure and compare performance after each change

4. **Adaptive Processing**
   - Implement a system that dynamically chooses CPU or GPU based on workload
   - For small videos, use CPU
   - For large videos or batch processing, use GPU

## 6. Timeline Estimation

1. **Phase 1: Batch Processing Implementation**
   - Estimated time: 1-2 weeks
   - Expected outcomes: Initial performance improvement for larger videos

2. **Phase 2: Memory Optimization**
   - Estimated time: 1-2 weeks
   - Expected outcomes: Reduced overhead, better scaling for larger videos

3. **Phase 3: Algorithm Adaptations**
   - Estimated time: 2-3 weeks
   - Expected outcomes: Significant improvement for motion detection

4. **Phase 4: OpenCV with CUDA Integration**
   - Estimated time: 1-2 weeks
   - Expected outcomes: Complete GPU acceleration pipeline

## Conclusion

By implementing these optimizations, we expect to achieve significant performance improvements for our GPU-accelerated steganography system, particularly for higher resolution videos and batch processing scenarios. The goal is to reach at least 2-3x speedup for large videos compared to the CPU implementation. 