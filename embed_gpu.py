#!/usr/bin/env python3
import cv2
import numpy as np
import pickle
import os
import time
import argparse
from dct_utils_gpu import *

def embed_bits(coeffs, bits, indices=(4, 5, 6, 7, 10, 11)):
    """Embed 3 bits into 6 coefficients using psychovisual thresholds"""
    # Target coefficients indices - default to (4,5,6,7,10,11) in middle frequency
    target_indices = list(indices)
    
    # Get psychovisual thresholds for each coefficient pair
    min_diff_1 = get_psychovisual_threshold(target_indices[0])
    min_diff_2 = get_psychovisual_threshold(target_indices[2])
    min_diff_3 = get_psychovisual_threshold(target_indices[4])
    
    # Embed first bit using coefficients 4 and 5
    if bits[0] == 1:
        # For a '1' bit, make sure coeff[4] > coeff[5]
        if coeffs[target_indices[0]] <= coeffs[target_indices[1]]:
            # Swap and ensure a minimum difference
            diff = abs(coeffs[target_indices[0]] - coeffs[target_indices[1]])
            if diff < min_diff_1:
                diff = min_diff_1
            coeffs[target_indices[0]] = coeffs[target_indices[1]] + diff
        else:
            # Already in the right order, but ensure minimum difference
            diff = coeffs[target_indices[0]] - coeffs[target_indices[1]]
            if diff < min_diff_1:
                coeffs[target_indices[0]] = coeffs[target_indices[1]] + min_diff_1
    else:
        # For a '0' bit, make sure coeff[4] < coeff[5]
        if coeffs[target_indices[0]] >= coeffs[target_indices[1]]:
            # Swap and ensure a minimum difference
            diff = abs(coeffs[target_indices[0]] - coeffs[target_indices[1]])
            if diff < min_diff_1:
                diff = min_diff_1
            coeffs[target_indices[0]] = coeffs[target_indices[1]] - diff
        else:
            # Already in the right order, but ensure minimum difference
            diff = coeffs[target_indices[1]] - coeffs[target_indices[0]]
            if diff < min_diff_1:
                coeffs[target_indices[0]] = coeffs[target_indices[1]] - min_diff_1
    
    # Embed second bit using coefficients 6 and 7
    if bits[1] == 1:
        # For a '1' bit, make sure coeff[6] > coeff[7]
        if coeffs[target_indices[2]] <= coeffs[target_indices[3]]:
            # Swap and ensure a minimum difference
            diff = abs(coeffs[target_indices[2]] - coeffs[target_indices[3]])
            if diff < min_diff_2:
                diff = min_diff_2
            coeffs[target_indices[2]] = coeffs[target_indices[3]] + diff
        else:
            # Already in the right order, but ensure minimum difference
            diff = coeffs[target_indices[2]] - coeffs[target_indices[3]]
            if diff < min_diff_2:
                coeffs[target_indices[2]] = coeffs[target_indices[3]] + min_diff_2
    else:
        # For a '0' bit, make sure coeff[6] < coeff[7]
        if coeffs[target_indices[2]] >= coeffs[target_indices[3]]:
            # Swap and ensure a minimum difference
            diff = abs(coeffs[target_indices[2]] - coeffs[target_indices[3]])
            if diff < min_diff_2:
                diff = min_diff_2
            coeffs[target_indices[2]] = coeffs[target_indices[3]] - diff
        else:
            # Already in the right order, but ensure minimum difference
            diff = coeffs[target_indices[3]] - coeffs[target_indices[2]]
            if diff < min_diff_2:
                coeffs[target_indices[2]] = coeffs[target_indices[3]] - min_diff_2
    
    # Embed third bit using coefficients 10 and 11
    if bits[2] == 1:
        # For a '1' bit, make sure coeff[10] > coeff[11]
        if coeffs[target_indices[4]] <= coeffs[target_indices[5]]:
            # Swap and ensure a minimum difference
            diff = abs(coeffs[target_indices[4]] - coeffs[target_indices[5]])
            if diff < min_diff_3:
                diff = min_diff_3
            coeffs[target_indices[4]] = coeffs[target_indices[5]] + diff
        else:
            # Already in the right order, but ensure minimum difference
            diff = coeffs[target_indices[4]] - coeffs[target_indices[5]]
            if diff < min_diff_3:
                coeffs[target_indices[4]] = coeffs[target_indices[5]] + min_diff_3
    else:
        # For a '0' bit, make sure coeff[10] < coeff[11]
        if coeffs[target_indices[4]] >= coeffs[target_indices[5]]:
            # Swap and ensure a minimum difference
            diff = abs(coeffs[target_indices[4]] - coeffs[target_indices[5]])
            if diff < min_diff_3:
                diff = min_diff_3
            coeffs[target_indices[4]] = coeffs[target_indices[5]] - diff
        else:
            # Already in the right order, but ensure minimum difference
            diff = coeffs[target_indices[5]] - coeffs[target_indices[4]]
            if diff < min_diff_3:
                coeffs[target_indices[4]] = coeffs[target_indices[5]] - min_diff_3
    
    return coeffs

def embed_message(video_path, message, output_path, locations_file, repetition_factor=5, format_preset="default", use_gpu=True):
    """Embed message into video using DCT-based steganography with GPU acceleration
    
    Args:
        video_path: Path to input video
        message: Message to embed
        output_path: Path to save output video
        locations_file: File to save embedding locations
        repetition_factor: Error correction repetition factor
        format_preset: Output format preset ("default", "lossless", "raw", "high_quality")
        use_gpu: Whether to use GPU acceleration if available
    """
    # Check if GPU acceleration is available
    from dct_utils_gpu import use_gpu as is_gpu_available
    gpu_available = use_gpu and is_gpu_available()
    
    if gpu_available:
        print("Using GPU acceleration")
    else:
        print("Using CPU processing (GPU not available or disabled)")
    
    start_time = time.time()
    
    # Convert message to binary with error correction (repeat each bit)
    binary_message_raw = ''.join(format(ord(char), '08b') for char in message)
    # Apply repetition code - repeat each bit for error correction
    binary_message = ''
    for bit in binary_message_raw:
        binary_message += bit * repetition_factor  # Repeat each bit
    
    message_length = len(binary_message)
    print(f"Original message length in bits: {len(binary_message_raw)}")
    print(f"Message length with error correction: {message_length}")
    
    # Save the original raw binary message for reference
    with open("original_binary_raw.txt", "w") as f:
        f.write(binary_message_raw)
    
    # Print the original message as binary for debugging
    print(f"Original message: {message}")
    print(f"Message as binary (raw): {binary_message_raw}")
    print(f"Message as binary (with error correction): {binary_message[:60]}...")
    
    # Print the first few bytes as binary and ASCII for verification
    if len(message) >= 3:
        first_bytes = []
        for i in range(min(3, len(message))):
            char = message[i]
            char_code = ord(char)
            binary = format(char_code, '08b')
            first_bytes.append(f"'{char}' ({char_code}) = {binary}")
        print(f"First 3 characters: {', '.join(first_bytes)}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open input video")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Format presets with optimal codec configurations for steganography
    format_presets = {
        "default": {
            "extension": ".mp4",
            "codecs": ['mp4v', 'XVID', 'MJPG', 'X264'],
            "params": {} 
        },
        "lossless": {
            "extension": ".avi", 
            "codecs": ['HFYU', 'FFVH', 'FFV1'],  # Huffman YUV, FFmpeg Huffman, FFmpeg lossless
            "params": {}
        },
        "raw": {
            "extension": ".avi",
            "codecs": ['IYUV', 'I420', 'Y800'],  # Uncompressed YUV formats
            "params": {}
        },
        "high_quality": {
            "extension": ".avi",
            "codecs": ['MJPG'],  # Motion JPEG, higher quality than MP4
            "params": {
                # Higher quality parameters for MJPG
                cv2.VIDEOWRITER_PROP_QUALITY: 95
            }
        }
    }
    
    # Get preset configuration
    preset = format_presets.get(format_preset, format_presets["default"])
    
    # Adjust output filename extension if needed
    output_root, output_ext = os.path.splitext(output_path)
    if output_ext.lower() != preset["extension"]:
        output_path = output_root + preset["extension"]
        print(f"Adjusted output path to match format: {output_path}")
    
    # Try codecs from the preset
    out = None
    used_codec = None
    
    for codec in preset["codecs"]:
        try:
            print(f"Trying codec: {codec}")
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Apply additional codec parameters if supported
            for param, value in preset["params"].items():
                out.set(param, value)
            
            if out.isOpened():
                used_codec = codec
                print(f"Successfully opened video writer with codec: {codec}")
                break
        except Exception as e:
            print(f"Failed to use codec {codec}: {str(e)}")
            if out is not None:
                out.release()
    
    if out is None or not out.isOpened():
        raise ValueError("Could not create output video with any available codec")
    
    # Store embedding locations
    embedding_locations = []
    bit_index = 0
    
    # Read all frames first
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    print(f"Read {len(frames)} frames")
    
    # Create a directory for quality analysis
    os.makedirs("quality_analysis", exist_ok=True)
    
    # Store PSNR values for each frame
    psnr_values = []
    
    # Process frames with motion analysis
    prev_frame = None
    dct_time = 0
    embed_time = 0
    motion_time = 0
    
    for frame_idx, frame in enumerate(frames, 1):  # Start frame numbers from 1
        # Track progress
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{len(frames)}, embedded {bit_index}/{message_length} bits")
        
        # Skip further processing if we've embedded all bits
        if bit_index >= message_length:
            # Write remaining frames without modification
            out.write(frame)
            continue
        
        # Convert frame to YCrCb
        frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = frame_ycrcb[:,:,0]
        
        # Save original Y channel for PSNR calculation
        original_y = y_channel.copy()
        
        # Only process blocks if we still have bits to embed
        if bit_index < message_length:
            # Use motion analysis to select blocks if previous frame exists
            motion_start = time.time()
            if prev_frame is not None and frame_idx > 1:
                # Get blocks with motion
                blocks = select_motion_blocks(prev_frame, frame, max_blocks=5, use_gpu=gpu_available)
                
                # If not enough motion blocks, get additional blocks based on frame index
                if len(blocks) < 3:
                    additional_blocks = select_blocks_for_frame(y_channel, frame_idx)
                    for block in additional_blocks:
                        if block not in blocks:
                            blocks.append(block)
                            if len(blocks) >= 5:
                                break
            else:
                # For first frame, use frame-based selection
                blocks = select_blocks_for_frame(y_channel, frame_idx)
            motion_time += time.time() - motion_start
            
            # Adjust DCT coefficient thresholds based on output format
            # Higher thresholds for more compressed formats
            threshold_multiplier = 1.0
            if used_codec in ['mp4v', 'X264', 'XVID']:
                threshold_multiplier = 1.5  # Increase thresholds for lossy formats
            elif used_codec in ['MJPG']:
                threshold_multiplier = 1.2  # Slightly increase for MJPG
            
            # Extract all blocks for batch processing
            block_data = []
            block_positions = []
            
            for x, y in blocks:
                if bit_index >= message_length:
                    break
                
                block = y_channel[y:y+8, x:x+8].astype(np.float32)
                block_data.append(block)
                block_positions.append((x, y))
            
            # Batch DCT processing using GPU if available
            dct_start = time.time()
            if gpu_available and len(block_data) > 0:
                dct_blocks = parallel_dct_batch(block_data, use_gpu=True)
            else:
                dct_blocks = [block_dct(block, use_gpu=False) for block in block_data]
            dct_time += time.time() - dct_start
            
            # Process each block
            embed_start = time.time()
            for i, ((x, y), dct_block) in enumerate(zip(block_positions, dct_blocks)):
                if bit_index >= message_length:
                    break
                
                # Convert to zigzag order
                zigzag = zigzag_scan(dct_block)
                
                # Calculate thresholds
                f, s = calculate_thresholds(zigzag, T=20 * threshold_multiplier)
                
                # Get next 3 bits
                remaining_bits = message_length - bit_index
                if remaining_bits >= 3:
                    bits = [int(binary_message[i]) for i in range(bit_index, bit_index + 3)]
                    bit_index += 3
                else:
                    # Handle remaining bits
                    bits = [int(binary_message[i]) for i in range(bit_index, message_length)]
                    bits.extend([0] * (3 - remaining_bits))
                    bit_index = message_length  # Set to message_length to stop further embedding
                
                # Embed bits
                modified_zigzag = embed_bits(zigzag, bits)
                
                # Convert back to block
                modified_dct = inverse_zigzag_scan(modified_zigzag)
                
                # Apply IDCT
                if gpu_available:
                    modified_block = block_idct(modified_dct, use_gpu=True)
                else:
                    modified_block = block_idct(modified_dct, use_gpu=False)
                
                # Update frame
                y_channel[y:y+8, x:x+8] = modified_block
                
                # Store location with correct frame number
                embedding_locations.append((frame_idx, x, y))
                
                if len(embedding_locations) <= 5 or len(embedding_locations) % 10 == 0:
                    print(f"Embedded at location: frame {frame_idx}, position ({x}, {y})")
            embed_time += time.time() - embed_start
        
        # Convert back to BGR
        frame_ycrcb[:,:,0] = y_channel
        modified_frame = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        # Calculate PSNR between original and modified Y channel
        psnr = calculate_psnr(original_y, y_channel)
        psnr_values.append(psnr)
        
        # Save key frames for visual comparison
        if frame_idx % 30 == 0 or frame_idx == 1:
            comparison = np.hstack((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                    cv2.cvtColor(modified_frame, cv2.COLOR_BGR2RGB)))
            cv2.imwrite(f"quality_analysis/frame_{frame_idx}_comparison.jpg", comparison)
        
        # Write frame
        out.write(modified_frame)
        
        # Update previous frame
        prev_frame = frame.copy()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Embedding complete. Embedded {bit_index} bits in {len(embedding_locations)} blocks")
    print(f"Output format: {preset['extension'][1:].upper()}, Codec: {used_codec}")
    
    # Print timing information
    print(f"\nPerformance statistics:")
    print(f"  Total processing time: {total_time:.2f} seconds")
    print(f"  DCT/IDCT operations: {dct_time:.2f} seconds ({dct_time/total_time*100:.1f}%)")
    print(f"  Embedding operations: {embed_time:.2f} seconds ({embed_time/total_time*100:.1f}%)")
    print(f"  Motion detection: {motion_time:.2f} seconds ({motion_time/total_time*100:.1f}%)")
    print(f"  Processing speed: {len(frames)/total_time:.2f} frames per second")
    
    # Print quality metrics
    if psnr_values:
        avg_psnr = np.mean(psnr_values)
        min_psnr = np.min(psnr_values)
        max_psnr = np.max(psnr_values)
        
        print(f"\nQuality metrics:")
        print(f"  Average PSNR: {avg_psnr:.2f} dB")
        print(f"  Minimum PSNR: {min_psnr:.2f} dB")
        print(f"  Maximum PSNR: {max_psnr:.2f} dB")
        
        # Save PSNR values to file
        with open("quality_analysis/psnr_values.txt", "w") as f:
            for i, psnr in enumerate(psnr_values, 1):
                f.write(f"Frame {i}: {psnr:.2f} dB\n")
    
    # Save embedding locations
    with open(locations_file, 'wb') as f:
        pickle.dump(embedding_locations, f)
    
    # Save original binary message for comparison
    ref_file = "original_binary.txt"
    with open(ref_file, 'w') as f:
        f.write(binary_message)
    print(f"Saved original binary message to {ref_file} for comparison")
    
    # Also save the mapping of which locations contain which bits
    with open("embedding_map.txt", "w") as f:
        for i, (frame_num, x, y) in enumerate(embedding_locations):
            bit_start = i * 3
            bit_end = min(bit_start + 3, message_length)
            if bit_end > bit_start:
                bits = binary_message[bit_start:bit_end]
                f.write(f"Location {i}: Frame {frame_num}, Position ({x}, {y}), Bits {bit_start}-{bit_end-1}: {bits}\n")
    
    # Cleanup
    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embed message in video using GPU-accelerated DCT-based steganography')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('message', help='Message to embed')
    parser.add_argument('output_video', help='Output video file path')
    parser.add_argument('locations_file', help='File to store embedding locations')
    parser.add_argument('--repetition', type=int, default=5, help='Repetition factor for error correction (default: 5)')
    parser.add_argument('--format', choices=['default', 'lossless', 'raw', 'high_quality'], default='default',
                        help='Output format preset (default: default)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    embed_message(args.input_video, args.message, args.output_video, args.locations_file, 
                  args.repetition, args.format, not args.no_gpu) 