import cv2
import numpy as np
import pickle
import os
from dct_utils import *

def extract_bits(coeffs, indices=(4, 5, 6, 7, 10, 11)):
    """Extract 3 bits from 6 coefficients using specified indices"""
    # Target coefficients indices - default to (4,5,6,7,10,11)
    target_indices = list(indices)
    
    bits = []
    
    # Extract first bit from first coefficient pair
    bits.append(1 if coeffs[target_indices[0]] > coeffs[target_indices[1]] else 0)
    
    # Extract second bit from second coefficient pair
    bits.append(1 if coeffs[target_indices[2]] > coeffs[target_indices[3]] else 0)
    
    # Extract third bit from third coefficient pair
    bits.append(1 if coeffs[target_indices[4]] > coeffs[target_indices[5]] else 0)
    
    return bits

def extract_message(video_path, locations_file, repetition_factor=5):
    """Extract message from video using DCT-based steganography"""
    # Load embedding locations
    with open(locations_file, 'rb') as f:
        embedding_locations = pickle.load(f)
    
    print(f"Loaded {len(embedding_locations)} embedding locations")
    
    # Print first few locations for debugging
    print("First 5 embedding locations:")
    for i, loc in enumerate(embedding_locations[:5]):
        print(f"  Location {i}: frame {loc[0]}, position ({loc[1]}, {loc[2]})")
    
    # Print some statistics about frame distribution
    frame_numbers = [loc[0] for loc in embedding_locations]
    unique_frames = sorted(set(frame_numbers))
    print(f"Unique frame numbers: {unique_frames}")
    print(f"Frame number distribution: {[(f, frame_numbers.count(f)) for f in unique_frames]}")
    
    # Get video format information
    _, ext = os.path.splitext(video_path)
    print(f"Video format: {ext.upper()}")
    
    # Try to open the video with multiple backends if needed
    cap = None
    backends = [cv2.CAP_ANY]  # Default backend
    
    # For AVI files, sometimes different backends work better
    if ext.lower() == '.avi':
        backends.extend([cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_MSMF])
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(video_path, backend)
            if cap.isOpened():
                print(f"Successfully opened video with backend ID: {backend}")
                break
        except Exception as e:
            print(f"Failed to open video with backend {backend}: {str(e)}")
            if cap is not None:
                cap.release()
    
    if not cap or not cap.isOpened():
        raise ValueError("Could not open input video with any available backend")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get codec information
    codec_numeric = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = chr(codec_numeric & 0xFF) + chr((codec_numeric >> 8) & 0xFF) + chr((codec_numeric >> 16) & 0xFF) + chr((codec_numeric >> 24) & 0xFF)
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames, Codec: {codec}")
    
    # Read all frames first
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    print(f"Read {len(frames)} frames")
    
    # Create directory for extraction analysis
    os.makedirs("extraction_analysis", exist_ok=True)
    
    # Extract bits
    extracted_bits = []
    processed_locations = 0
    
    # Sort locations by frame number for more efficient reading
    embedding_locations.sort(key=lambda x: x[0])
    
    # Process each embedding location
    extraction_errors = 0
    
    for frame_num, x, y in embedding_locations:
        if frame_num > len(frames):
            print(f"Warning: Frame number {frame_num} exceeds video length")
            extraction_errors += 1
            continue
            
        if processed_locations % 10 == 0:
            print(f"Processing location {processed_locations}/{len(embedding_locations)}")
            print(f"Processing frame {frame_num}")
        
        # Get frame (frame_num is 1-based)
        frame_idx = frame_num - 1  # Convert to 0-based index
        if frame_idx < 0 or frame_idx >= len(frames):
            print(f"Warning: Invalid frame index {frame_idx} for frame number {frame_num}")
            extraction_errors += 1
            continue
            
        frame = frames[frame_idx]
        
        # Convert frame to YCrCb
        frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = frame_ycrcb[:,:,0]
        
        # Extract 8x8 block
        block = y_channel[y:y+8, x:x+8].astype(np.float32)
        
        # Apply DCT
        dct_block = block_dct(block)
        
        # Convert to zigzag order
        zigzag = zigzag_scan(dct_block)
        
        # Save DCT coefficients for analysis
        if processed_locations < 5:
            coeffs_str = " ".join([f"{c:.2f}" for c in zigzag[4:12]])
            with open(f"extraction_analysis/block_{processed_locations}_coeffs.txt", "w") as f:
                f.write(f"Frame {frame_num}, Position ({x}, {y})\n")
                f.write(f"DCT Coefficients: {coeffs_str}\n")
        
        # Extract bits
        bits = extract_bits(zigzag)
        extracted_bits.extend(bits)
        
        # Debug: show extracted bits for first few blocks
        if processed_locations < 5:
            print(f"  Block at ({x}, {y}): extracted bits {bits}")
            # Save the block image
            block_img = y_channel[y:y+8, x:x+8].astype(np.uint8)
            cv2.imwrite(f"extraction_analysis/block_{processed_locations}.png", block_img)
        
        processed_locations += 1
    
    print(f"Extracted {len(extracted_bits)} bits with {extraction_errors} errors during extraction")
    
    # Save extracted bits to file for comparison
    extracted_bits_str = ''.join(map(str, extracted_bits))
    with open("extracted_binary.txt", "w") as f:
        f.write(extracted_bits_str)
    
    # Apply error correction - repetition code
    corrected_bits = []
    if len(extracted_bits) >= repetition_factor:
        # Process in groups based on repetition factor
        for i in range(0, len(extracted_bits), repetition_factor):
            if i + repetition_factor <= len(extracted_bits):
                group = extracted_bits[i:i+repetition_factor]
                # Majority vote
                bit = 1 if sum(group) >= repetition_factor // 2 + 1 else 0
                corrected_bits.append(bit)
    
    # Save corrected bits to file
    corrected_bits_str = ''.join(map(str, corrected_bits))
    with open("corrected_binary.txt", "w") as f:
        f.write(corrected_bits_str)
    
    valid_bit_length = (len(corrected_bits) // 8) * 8  # Ensure we have complete bytes
    valid_bits = corrected_bits[:valid_bit_length]
    
    print(f"After error correction: {len(corrected_bits)} bits, valid bytes: {valid_bit_length} bits")
    
    # Compare with original if available
    try:
        with open("original_binary_raw.txt", "r") as f:
            original_binary = f.read().strip()
        
        print(f"Original binary length: {len(original_binary)}")
        print(f"Corrected binary length: {len(corrected_bits_str)}")
        
        # Compare bit by bit
        if len(original_binary) == len(corrected_bits_str):
            diff_count = sum(1 for a, b in zip(original_binary, corrected_bits_str) if a != b)
            diff_percent = (diff_count / len(original_binary)) * 100
            print(f"Bits differ in {diff_count} positions ({diff_percent:.2f}%)")
            
            # Calculate Normalized Correlation (NC)
            orig_bits = [int(b) for b in original_binary]
            corr_bits = [int(b) for b in corrected_bits_str]
            
            nc = sum(a * b for a, b in zip(orig_bits, corr_bits)) / sum(a ** 2 for a in orig_bits)
            print(f"Normalized Correlation (NC): {nc:.4f}")
            
            # Show the first few differences
            print("First differences:")
            diff_shown = 0
            for i, (a, b) in enumerate(zip(original_binary, corrected_bits_str)):
                if a != b and diff_shown < 5:  # Show first 5 differences
                    print(f"Position {i}: Original={a}, Extracted={b}")
                    diff_shown += 1
            
            # Save a bit error map for analysis
            with open("extraction_analysis/bit_error_map.txt", "w") as f:
                for i, (a, b) in enumerate(zip(original_binary, corrected_bits_str)):
                    if a != b:
                        f.write(f"Error at position {i}: Original={a}, Extracted={b}\n")
        else:
            print("Cannot compare bit by bit due to length mismatch")
            
        # Show the original bytes vs extracted bytes
        print("\nOriginal vs Extracted (first 3 bytes):")
        for i in range(0, min(24, len(original_binary)), 8):
            orig_byte = original_binary[i:i+8]
            extr_byte = corrected_bits_str[i:i+8] if i+8 <= len(corrected_bits_str) else "incomplete"
            
            if orig_byte and len(orig_byte) == 8:
                orig_char = chr(int(orig_byte, 2))
                orig_val = int(orig_byte, 2)
            else:
                orig_char = "?"
                orig_val = -1
                
            if extr_byte and len(extr_byte) == 8:
                extr_char = chr(int(extr_byte, 2))
                extr_val = int(extr_byte, 2)
            else:
                extr_char = "?"
                extr_val = -1
                
            print(f"Byte {i//8}: Original='{orig_char}'({orig_val}) {orig_byte}, Extracted='{extr_char}'({extr_val}) {extr_byte}")
    except FileNotFoundError:
        print("Original binary file not found for comparison")
    
    # Convert to message
    message = ''
    for i in range(0, len(valid_bits), 8):
        if i + 8 <= len(valid_bits):
            byte = valid_bits[i:i+8]
            byte_str = ''.join(map(str, byte))
            char_code = int(byte_str, 2)
            try:
                char = chr(char_code)
                message += char
            except:
                print(f"Error with character code {char_code}")
                message += '?'
    
    # Cleanup
    cap.release()
    
    return message

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract message from video using DCT-based steganography')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('locations_file', help='File containing embedding locations')
    parser.add_argument('--repetition', type=int, default=5, help='Repetition factor for error correction (default: 5)')
    
    args = parser.parse_args()
    
    message = extract_message(args.input_video, args.locations_file, args.repetition)
    print("Extracted message:", message) 