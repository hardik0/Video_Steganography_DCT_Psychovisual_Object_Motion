import numpy as np
import cv2
import sys
from scipy.fftpack import dct, idct

# Try to import cupy, but handle case when it's not available
CUPY_AVAILABLE = False
try:
    print("Trying to import cupy...")
    import cupy as cp
    
    # Try to import fft functions from the correct module
    try:
        # Modern cupy versions use scipy.fft
        from cupyx.scipy.fft import dct as cuda_dct, idct as cuda_idct
        print("Using cupyx.scipy.fft for DCT/IDCT operations")
    except ImportError:
        # Older versions might use fftpack
        from cupyx.scipy.fftpack import dct as cuda_dct, idct as cuda_idct
        print("Using cupyx.scipy.fftpack for DCT/IDCT operations")
        
    CUPY_AVAILABLE = True
    print(f"CuPy successfully imported (version {cp.__version__})")
except ImportError as e:
    print(f"CuPy is not available: {str(e)}")
    print(f"Python path: {sys.path}")
    print("GPU acceleration will be disabled")
    # Create dummy functions that will not be used (just to avoid errors)
    cp = None
    cuda_dct = None
    cuda_idct = None
except Exception as e:
    print(f"Error importing cupy: {str(e)}")
    print("GPU acceleration will be disabled")
    cp = None
    cuda_dct = None
    cuda_idct = None

# Check if CUDA is available via cupy
CUDA_AVAILABLE = False
if CUPY_AVAILABLE:
    try:
        # Try to create a simple array on the GPU to check if CUDA works
        test_array = cp.array([1, 2, 3])
        test_result = test_array + test_array  # Test computation
        
        # Test DCT/IDCT operations
        test_dct = cp.ones((8, 8), dtype=cp.float32)
        dct_result = cuda_dct(test_dct, norm='ortho')
        idct_result = cuda_idct(dct_result, norm='ortho')
        
        CUDA_AVAILABLE = True
        
        # Get device info
        device_props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = device_props['name'].decode()
        total_memory = device_props['totalGlobalMem'] / (1024**3)
        
        print(f"CUDA is available with {cp.cuda.runtime.getDeviceCount()} device(s)")
        print(f"Using GPU: {device_name} with {total_memory:.2f} GB memory")
    except Exception as e:
        print(f"CUDA initialization error: {str(e)}")
        print("CUDA is available but not working properly")
        CUDA_AVAILABLE = False
else:
    print("CUDA is not available (CuPy not installed)")

# Try to also check OpenCV CUDA support (optional)
try:
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    if cuda_devices > 0:
        print(f"OpenCV CUDA support: {cuda_devices} device(s) detected")
    else:
        print("OpenCV was not built with CUDA support")
except:
    print("OpenCV CUDA support check failed")

def use_gpu():
    """Return whether GPU acceleration is available"""
    gpu_available = CUDA_AVAILABLE and CUPY_AVAILABLE
    print(f"GPU acceleration {'enabled' if gpu_available else 'disabled'}")
    return gpu_available

def block_dct(block, use_gpu=False):
    """Apply 2D DCT to an 8x8 block"""
    if use_gpu and CUDA_AVAILABLE and CUPY_AVAILABLE:
        # Transfer to GPU
        try:
            block_gpu = cp.asarray(block)
            # Apply DCT
            dct_result = cuda_dct(cuda_dct(block_gpu.T, norm='ortho').T, norm='ortho')
            # Transfer back to CPU
            return cp.asnumpy(dct_result)
        except Exception as e:
            print(f"GPU DCT error: {str(e)}, falling back to CPU")
            return dct(dct(block.T, norm='ortho').T, norm='ortho')
    else:
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

def block_idct(block, use_gpu=False):
    """Apply 2D IDCT to an 8x8 block"""
    if use_gpu and CUDA_AVAILABLE and CUPY_AVAILABLE:
        # Transfer to GPU
        try:
            block_gpu = cp.asarray(block)
            # Apply IDCT
            idct_result = cuda_idct(cuda_idct(block_gpu.T, norm='ortho').T, norm='ortho')
            # Transfer back to CPU
            return cp.asnumpy(idct_result)
        except Exception as e:
            print(f"GPU IDCT error: {str(e)}, falling back to CPU")
            return idct(idct(block.T, norm='ortho').T, norm='ortho')
    else:
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

def zigzag_scan(block):
    """Convert 8x8 block to zigzag order"""
    zigzag = np.zeros(64)
    zigzag[0] = block[0, 0]
    zigzag[1] = block[0, 1]
    zigzag[2] = block[1, 0]
    zigzag[3] = block[2, 0]
    zigzag[4] = block[1, 1]
    zigzag[5] = block[0, 2]
    zigzag[6] = block[0, 3]
    zigzag[7] = block[1, 2]
    zigzag[8] = block[2, 1]
    zigzag[9] = block[3, 0]
    zigzag[10] = block[4, 0]
    zigzag[11] = block[3, 1]
    zigzag[12] = block[2, 2]
    zigzag[13] = block[1, 3]
    zigzag[14] = block[0, 4]
    zigzag[15] = block[0, 5]
    zigzag[16] = block[1, 4]
    zigzag[17] = block[2, 3]
    zigzag[18] = block[3, 2]
    zigzag[19] = block[4, 1]
    zigzag[20] = block[5, 0]
    zigzag[21] = block[6, 0]
    zigzag[22] = block[5, 1]
    zigzag[23] = block[4, 2]
    zigzag[24] = block[3, 3]
    zigzag[25] = block[2, 4]
    zigzag[26] = block[1, 5]
    zigzag[27] = block[0, 6]
    zigzag[28] = block[0, 7]
    zigzag[29] = block[1, 6]
    zigzag[30] = block[2, 5]
    zigzag[31] = block[3, 4]
    zigzag[32] = block[4, 3]
    zigzag[33] = block[5, 2]
    zigzag[34] = block[6, 1]
    zigzag[35] = block[7, 0]
    zigzag[36] = block[7, 1]
    zigzag[37] = block[6, 2]
    zigzag[38] = block[5, 3]
    zigzag[39] = block[4, 4]
    zigzag[40] = block[3, 5]
    zigzag[41] = block[2, 6]
    zigzag[42] = block[1, 7]
    zigzag[43] = block[2, 7]
    zigzag[44] = block[3, 6]
    zigzag[45] = block[4, 5]
    zigzag[46] = block[5, 4]
    zigzag[47] = block[6, 3]
    zigzag[48] = block[7, 2]
    zigzag[49] = block[7, 3]
    zigzag[50] = block[6, 4]
    zigzag[51] = block[5, 5]
    zigzag[52] = block[4, 6]
    zigzag[53] = block[3, 7]
    zigzag[54] = block[4, 7]
    zigzag[55] = block[5, 6]
    zigzag[56] = block[6, 5]
    zigzag[57] = block[7, 4]
    zigzag[58] = block[7, 5]
    zigzag[59] = block[6, 6]
    zigzag[60] = block[5, 7]
    zigzag[61] = block[6, 7]
    zigzag[62] = block[7, 6]
    zigzag[63] = block[7, 7]
    return zigzag

def inverse_zigzag_scan(zigzag):
    """Convert zigzag order back to 8x8 block"""
    block = np.zeros((8, 8))
    block[0, 0] = zigzag[0]
    block[0, 1] = zigzag[1]
    block[1, 0] = zigzag[2]
    block[2, 0] = zigzag[3]
    block[1, 1] = zigzag[4]
    block[0, 2] = zigzag[5]
    block[0, 3] = zigzag[6]
    block[1, 2] = zigzag[7]
    block[2, 1] = zigzag[8]
    block[3, 0] = zigzag[9]
    block[4, 0] = zigzag[10]
    block[3, 1] = zigzag[11]
    block[2, 2] = zigzag[12]
    block[1, 3] = zigzag[13]
    block[0, 4] = zigzag[14]
    block[0, 5] = zigzag[15]
    block[1, 4] = zigzag[16]
    block[2, 3] = zigzag[17]
    block[3, 2] = zigzag[18]
    block[4, 1] = zigzag[19]
    block[5, 0] = zigzag[20]
    block[6, 0] = zigzag[21]
    block[5, 1] = zigzag[22]
    block[4, 2] = zigzag[23]
    block[3, 3] = zigzag[24]
    block[2, 4] = zigzag[25]
    block[1, 5] = zigzag[26]
    block[0, 6] = zigzag[27]
    block[0, 7] = zigzag[28]
    block[1, 6] = zigzag[29]
    block[2, 5] = zigzag[30]
    block[3, 4] = zigzag[31]
    block[4, 3] = zigzag[32]
    block[5, 2] = zigzag[33]
    block[6, 1] = zigzag[34]
    block[7, 0] = zigzag[35]
    block[7, 1] = zigzag[36]
    block[6, 2] = zigzag[37]
    block[5, 3] = zigzag[38]
    block[4, 4] = zigzag[39]
    block[3, 5] = zigzag[40]
    block[2, 6] = zigzag[41]
    block[1, 7] = zigzag[42]
    block[2, 7] = zigzag[43]
    block[3, 6] = zigzag[44]
    block[4, 5] = zigzag[45]
    block[5, 4] = zigzag[46]
    block[6, 3] = zigzag[47]
    block[7, 2] = zigzag[48]
    block[7, 3] = zigzag[49]
    block[6, 4] = zigzag[50]
    block[5, 5] = zigzag[51]
    block[4, 6] = zigzag[52]
    block[3, 7] = zigzag[53]
    block[4, 7] = zigzag[54]
    block[5, 6] = zigzag[55]
    block[6, 5] = zigzag[56]
    block[7, 4] = zigzag[57]
    block[7, 5] = zigzag[58]
    block[6, 6] = zigzag[59]
    block[5, 7] = zigzag[60]
    block[6, 7] = zigzag[61]
    block[7, 6] = zigzag[62]
    block[7, 7] = zigzag[63]
    return block

def calculate_thresholds(coeffs, T=20):
    """Calculate thresholds f and s based on coefficient magnitudes"""
    f = np.mean(np.abs(coeffs)) + T
    s = f / 2
    return f, s

def detect_motion_gpu(prev_frame, curr_frame, threshold=30):
    """GPU-accelerated motion detection between frames"""
    # Fall back to CPU implementation since we can't be sure OpenCV has CUDA
    return detect_motion(prev_frame, curr_frame, threshold)

def detect_motion(prev_frame, curr_frame, threshold=30):
    """Detect motion between two frames using absolute difference (CPU version)"""
    if prev_frame is None or curr_frame is None:
        return None
    
    # Convert frames to grayscale if they are not already
    if len(prev_frame.shape) > 2:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
    
    if len(curr_frame.shape) > 2:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame
    
    # Calculate absolute difference between frames
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Threshold the difference to identify motion regions
    _, motion_mask = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    
    return motion_mask

def calculate_motion_magnitude(motion_mask, block_x, block_y, block_size=8):
    """Calculate the motion magnitude within a block"""
    if motion_mask is None:
        return 0
    
    # Extract the block from motion mask
    block = motion_mask[block_y:block_y+block_size, block_x:block_x+block_size]
    
    # Calculate the percentage of motion pixels in the block
    motion_percentage = np.sum(block > 0) / (block_size * block_size)
    
    return motion_percentage

def select_motion_blocks(prev_frame, curr_frame, max_blocks=10, block_size=8, motion_threshold=0.3, use_gpu=False):
    """Select blocks based on motion between frames"""
    if prev_frame is None or curr_frame is None:
        return []
    
    # Detect motion using CPU implementation (more reliable across systems)
    motion_mask = detect_motion(prev_frame, curr_frame)
        
    if motion_mask is None:
        return []
    
    # Find candidate blocks
    height, width = curr_frame.shape[:2]
    candidate_blocks = []
    
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            motion_magnitude = calculate_motion_magnitude(motion_mask, x, y, block_size)
            
            # If motion magnitude exceeds threshold, add to candidates
            if motion_magnitude > motion_threshold:
                candidate_blocks.append((x, y, motion_magnitude))
    
    # Sort by motion magnitude (descending)
    candidate_blocks.sort(key=lambda b: b[2], reverse=True)
    
    # Return top N blocks with highest motion
    top_blocks = [(x, y) for x, y, _ in candidate_blocks[:max_blocks]]
    
    # If not enough motion blocks, fill with regular blocks
    if len(top_blocks) < max_blocks:
        remaining_blocks = max_blocks - len(top_blocks)
        regular_blocks = select_blocks(curr_frame, block_size)
        
        # Add regular blocks that are not already in top_blocks
        for block in regular_blocks:
            if block not in top_blocks and len(top_blocks) < max_blocks:
                top_blocks.append(block)
    
    return top_blocks[:max_blocks]

def select_blocks(frame, block_size=8):
    """Select blocks for embedding based on simplified selection"""
    height, width = frame.shape[:2]
    blocks = []
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            # Simplified block selection - using every 4th block
            if (x + y) % (block_size * 4) == 0:
                blocks.append((x, y))
    return blocks

def select_blocks_for_frame(frame, frame_idx, block_size=8, max_blocks_per_frame=3):
    """Select blocks for embedding based on frame index to distribute across frames"""
    height, width = frame.shape[:2]
    blocks = []
    
    # Use frame index to create different offsets for different frames
    offset_x = (frame_idx * 32) % width
    offset_y = (frame_idx * 16) % height
    
    # Limit the number of blocks per frame to distribute the message
    count = 0
    for y in range(offset_y, height - block_size + 1, block_size * 4):
        for x in range(offset_x, width - block_size + 1, block_size * 4):
            if count < max_blocks_per_frame:
                blocks.append((x, y))
                count += 1
            else:
                return blocks
    
    return blocks

def get_psychovisual_threshold(block_idx):
    """Get psychovisual threshold for DCT coefficients based on human visual perception"""
    # Psychovisual thresholds for different DCT coefficient positions
    # These values are based on JPEG quantization tables which are derived from human visual perception
    if block_idx in [4, 5]:  # First pair (4,5)
        return 35.0  # Higher threshold for these coefficients
    elif block_idx in [6, 7]:  # Second pair (6,7)
        return 45.0  # Even higher threshold for these coefficients
    elif block_idx in [10, 11]:  # Third pair (10,11)
        return 40.0  # High threshold for these coefficients
    else:
        return 30.0  # Default threshold

def calculate_psnr(original, modified):
    """Calculate Peak Signal-to-Noise Ratio between two images"""
    if original.shape != modified.shape:
        raise ValueError("Input images must have the same dimensions")
    
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def parallel_dct_batch(blocks, use_gpu=False):
    """Apply DCT to a batch of blocks in parallel using GPU if available"""
    if not use_gpu or not CUDA_AVAILABLE or not CUPY_AVAILABLE:
        # Fall back to serial processing
        return [block_dct(block, False) for block in blocks]
    
    # Prepare batch for GPU processing
    if len(blocks) == 0:
        return []
    
    # Stack all blocks into a 3D array
    batch_np = np.stack(blocks)
    
    # Transfer to GPU
    batch_gpu = cp.asarray(batch_np)
    
    # Apply DCT to all blocks (first on rows, then on columns)
    # We need to loop, but it's still faster than transferring each block individually
    result_gpu = cp.zeros_like(batch_gpu)
    for i in range(len(blocks)):
        result_gpu[i] = cuda_dct(cuda_dct(batch_gpu[i].T, norm='ortho').T, norm='ortho')
    
    # Transfer back to CPU
    result_np = cp.asnumpy(result_gpu)
    
    # Split back into individual blocks
    return [result_np[i] for i in range(len(blocks))]

def parallel_idct_batch(blocks, use_gpu=False):
    """Apply IDCT to a batch of blocks in parallel using GPU if available"""
    if not use_gpu or not CUDA_AVAILABLE or not CUPY_AVAILABLE:
        # Fall back to serial processing
        return [block_idct(block, False) for block in blocks]
    
    # Prepare batch for GPU processing
    if len(blocks) == 0:
        return []
    
    # Stack all blocks into a 3D array
    batch_np = np.stack(blocks)
    
    # Transfer to GPU
    batch_gpu = cp.asarray(batch_np)
    
    # Apply IDCT to all blocks (first on rows, then on columns)
    result_gpu = cp.zeros_like(batch_gpu)
    for i in range(len(blocks)):
        result_gpu[i] = cuda_idct(cuda_idct(batch_gpu[i].T, norm='ortho').T, norm='ortho')
    
    # Transfer back to CPU
    result_np = cp.asnumpy(result_gpu)
    
    # Split back into individual blocks
    return [result_np[i] for i in range(len(blocks))] 