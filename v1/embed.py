import cv2
import numpy as np
import pickle
from dct_utils import *

def embed_bits(coeffs, bits, f, s):
    """Embed 3 bits into 6 coefficients using the specified thresholds"""
    # Target coefficients indices (4,5,6,7,10,11)
    target_indices = [4, 5, 6, 7, 10, 11]
    
    # Use a consistent threshold for swapping
    min_diff = 30.0  # Increased from 15.0 for even better robustness
    
    # Embed first bit using coefficients 4 and 5
    if bits[0] == 1:
        # For a '1' bit, make sure coeff[4] > coeff[5]
        if coeffs[4] <= coeffs[5]:
            # Swap and ensure a minimum difference
            diff = abs(coeffs[4] - coeffs[5])
            if diff < min_diff:
                diff = min_diff
            coeffs[4] = coeffs[5] + diff
        else:
            # Already in the right order, but ensure minimum difference
            diff = coeffs[4] - coeffs[5]
            if diff < min_diff:
                coeffs[4] = coeffs[5] + min_diff
    else:
        # For a '0' bit, make sure coeff[4] < coeff[5]
        if coeffs[4] >= coeffs[5]:
            # Swap and ensure a minimum difference
            diff = abs(coeffs[4] - coeffs[5])
            if diff < min_diff:
                diff = min_diff
            coeffs[4] = coeffs[5] - diff
        else:
            # Already in the right order, but ensure minimum difference
            diff = coeffs[5] - coeffs[4]
            if diff < min_diff:
                coeffs[4] = coeffs[5] - min_diff
    
    # Embed second bit using coefficients 6 and 7
    if bits[1] == 1:
        # For a '1' bit, make sure coeff[6] > coeff[7]
        if coeffs[6] <= coeffs[7]:
            # Swap and ensure a minimum difference
            diff = abs(coeffs[6] - coeffs[7])
            if diff < min_diff:
                diff = min_diff
            coeffs[6] = coeffs[7] + diff
        else:
            # Already in the right order, but ensure minimum difference
            diff = coeffs[6] - coeffs[7]
            if diff < min_diff:
                coeffs[6] = coeffs[7] + min_diff
    else:
        # For a '0' bit, make sure coeff[6] < coeff[7]
        if coeffs[6] >= coeffs[7]:
            # Swap and ensure a minimum difference
            diff = abs(coeffs[6] - coeffs[7])
            if diff < min_diff:
                diff = min_diff
            coeffs[6] = coeffs[7] - diff
        else:
            # Already in the right order, but ensure minimum difference
            diff = coeffs[7] - coeffs[6]
            if diff < min_diff:
                coeffs[6] = coeffs[7] - min_diff
    
    # Embed third bit using coefficients 10 and 11
    if bits[2] == 1:
        # For a '1' bit, make sure coeff[10] > coeff[11]
        if coeffs[10] <= coeffs[11]:
            # Swap and ensure a minimum difference
            diff = abs(coeffs[10] - coeffs[11])
            if diff < min_diff:
                diff = min_diff
            coeffs[10] = coeffs[11] + diff
        else:
            # Already in the right order, but ensure minimum difference
            diff = coeffs[10] - coeffs[11]
            if diff < min_diff:
                coeffs[10] = coeffs[11] + min_diff
    else:
        # For a '0' bit, make sure coeff[10] < coeff[11]
        if coeffs[10] >= coeffs[11]:
            # Swap and ensure a minimum difference
            diff = abs(coeffs[10] - coeffs[11])
            if diff < min_diff:
                diff = min_diff
            coeffs[10] = coeffs[11] - diff
        else:
            # Already in the right order, but ensure minimum difference
            diff = coeffs[11] - coeffs[10]
            if diff < min_diff:
                coeffs[10] = coeffs[11] - min_diff
    
    return coeffs

def embed_message(video_path, message, output_path, locations_file):
    """Embed message into video using DCT-based steganography"""
    # Convert message to binary with simple error correction (repeat each bit 3 times)
    binary_message_raw = ''.join(format(ord(char), '08b') for char in message)
    # Apply repetition code - repeat each bit 5 times for better error correction
    binary_message = ''
    for bit in binary_message_raw:
        binary_message += bit * 5  # Repeat each bit 5 times
    
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
    
    # Try different codecs
    codecs = ['mp4v', 'XVID', 'MJPG', 'X264']
    out = None
    
    for codec in codecs:
        try:
            print(f"Trying codec: {codec}")
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
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
    
    # Process frames
    for frame_idx, frame in enumerate(frames, 1):  # Start frame numbers from 1
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{len(frames)}, embedded {bit_index}/{message_length} bits")
            
        # Convert frame to YCrCb
        frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = frame_ycrcb[:,:,0]
        
        # Only process blocks if we still have bits to embed
        if bit_index < message_length:
            # Select blocks for embedding - use the new frame-aware function
            blocks = select_blocks_for_frame(y_channel, frame_idx)
            
            for x, y in blocks:
                if bit_index >= message_length:
                    break
                    
                # Extract 8x8 block
                block = y_channel[y:y+8, x:x+8].astype(np.float32)
                
                # Apply DCT
                dct_block = block_dct(block)
                
                # Convert to zigzag order
                zigzag = zigzag_scan(dct_block)
                
                # Calculate thresholds
                f, s = calculate_thresholds(zigzag)
                
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
                modified_zigzag = embed_bits(zigzag, bits, f, s)
                
                # Convert back to block
                modified_dct = inverse_zigzag_scan(modified_zigzag)
                
                # Apply IDCT
                modified_block = block_idct(modified_dct)
                
                # Update frame
                y_channel[y:y+8, x:x+8] = modified_block
                
                # Store location with correct frame number
                embedding_locations.append((frame_idx, x, y))
                
                if len(embedding_locations) <= 5 or len(embedding_locations) % 10 == 0:
                    print(f"Embedded at location: frame {frame_idx}, position ({x}, {y})")
        
        # Convert back to BGR
        frame_ycrcb[:,:,0] = y_channel
        modified_frame = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        # Write frame
        out.write(modified_frame)
    
    print(f"Embedding complete. Embedded {bit_index} bits in {len(embedding_locations)} blocks")
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Embed message in video using DCT-based steganography')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('message', help='Message to embed')
    parser.add_argument('output_video', help='Output video file path')
    parser.add_argument('locations_file', help='File to store embedding locations')
    
    args = parser.parse_args()
    
    embed_message(args.input_video, args.message, args.output_video, args.locations_file) 