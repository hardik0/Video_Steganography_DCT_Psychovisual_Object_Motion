#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from tabulate import tabulate
from dct_utils import calculate_psnr

def get_video_info(video_path):
    """Get basic information about a video file"""
    if not os.path.exists(video_path):
        return None
    
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'path': video_path,
            'file_size': f"{file_size:.2f} MB",
            'error': "Could not open video"
        }
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get codec information
    codec_numeric = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = chr(codec_numeric & 0xFF) + chr((codec_numeric >> 8) & 0xFF) + chr((codec_numeric >> 16) & 0xFF) + chr((codec_numeric >> 24) & 0xFF)
    
    # Release video
    cap.release()
    
    return {
        'path': video_path,
        'file_size': f"{file_size:.2f} MB",
        'dimensions': f"{width}x{height}",
        'fps': fps,
        'frames': frames,
        'codec': codec
    }

def compare_videos(video_files, output_dir="comparison"):
    """Compare multiple video files and generate comparison frames"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get video info for each file
    video_info = [get_video_info(v) for v in video_files]
    valid_videos = [v for v in video_info if v and 'error' not in v]
    
    # Print comparison table
    headers = ["File", "Size", "Dimensions", "FPS", "Frames", "Codec"]
    table_data = []
    
    for info in valid_videos:
        table_data.append([
            os.path.basename(info['path']),
            info['file_size'],
            info['dimensions'],
            f"{info['fps']:.2f}",
            info['frames'],
            info['codec']
        ])
    
    print("\nVideo File Comparison:\n")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Extract sample frames from each video at the same positions
    caps = []
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            caps.append((cap, os.path.basename(video_path)))
    
    if not caps:
        print("No valid videos to compare")
        return
    
    # Compare frames at multiple positions
    frame_positions = [0, 0.25, 0.5, 0.75]  # Beginning, 25%, 50%, 75%
    
    for pos in frame_positions:
        pos_frames = []
        
        for cap, name in caps:
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_frame = int(total_frames * pos)
            
            # Set position
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            if ret:
                # Add filename to frame
                cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pos_frames.append(frame)
        
        # Create comparison image
        if pos_frames:
            # Resize all frames to the same height
            height = min(frame.shape[0] for frame in pos_frames)
            resized_frames = []
            
            for frame in pos_frames:
                aspect = frame.shape[1] / frame.shape[0]
                new_width = int(height * aspect)
                resized = cv2.resize(frame, (new_width, height))
                resized_frames.append(resized)
            
            # Concatenate horizontally
            comparison = np.hstack(resized_frames)
            
            # Save comparison
            output_path = os.path.join(output_dir, f"frame_comparison_{int(pos*100)}percent.jpg")
            cv2.imwrite(output_path, comparison)
            print(f"Saved comparison at {pos*100:.0f}% position: {output_path}")
    
    # Release all captures
    for cap, _ in caps:
        cap.release()
    
    print("\nComparison complete! Check the output directory for visual comparisons.")

def calculate_video_quality(original_video, steganographic_videos):
    """Calculate quality metrics for steganographic videos compared to original"""
    print("\nCalculating video quality metrics...")
    
    # Open original video
    orig_cap = cv2.VideoCapture(original_video)
    if not orig_cap.isOpened():
        print(f"Could not open original video: {original_video}")
        return
    
    # Store steganographic video captures
    stego_caps = []
    for video_path in steganographic_videos:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            stego_caps.append((cap, os.path.basename(video_path)))
        else:
            print(f"Could not open steganographic video: {video_path}")
    
    # Calculate PSNR for sample frames
    sample_frames = [0, 30, 60, 90, 120]  # Sample at specific frame numbers
    results = {name: [] for _, name in stego_caps}
    
    for frame_num in sample_frames:
        # Read original frame
        orig_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, orig_frame = orig_cap.read()
        
        if not ret:
            continue
        
        # Convert to YCrCb and extract Y channel
        orig_ycrcb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2YCrCb)
        orig_y = orig_ycrcb[:,:,0]
        
        # Compare with each steganographic video
        for cap, name in stego_caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, stego_frame = cap.read()
            
            if not ret:
                continue
            
            # Convert to YCrCb and extract Y channel
            stego_ycrcb = cv2.cvtColor(stego_frame, cv2.COLOR_BGR2YCrCb)
            stego_y = stego_ycrcb[:,:,0]
            
            # Calculate PSNR
            psnr = calculate_psnr(orig_y, stego_y)
            results[name].append(psnr)
    
    # Print results
    print("\nPSNR Values (Higher is better):")
    headers = ["Frame"] + [name for _, name in stego_caps]
    table_data = []
    
    for i, frame_num in enumerate(sample_frames):
        row = [f"Frame {frame_num}"]
        for _, name in stego_caps:
            if i < len(results[name]):
                if np.isinf(results[name][i]):
                    row.append("Perfect Match")
                else:
                    row.append(f"{results[name][i]:.2f} dB")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Add average row
    avg_row = ["Average"]
    for _, name in stego_caps:
        if results[name]:
            # Replace infinity with a high value (100) for calculation
            finite_values = [v if not np.isinf(v) else 100.0 for v in results[name]]
            avg = np.mean(finite_values)
            if all(np.isinf(v) for v in results[name]):
                avg_row.append("Perfect Match")
            else:
                avg_row.append(f"{avg:.2f} dB")
        else:
            avg_row.append("N/A")
    table_data.append(avg_row)
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Generate bar chart of average PSNR
    avg_psnr = []
    labels = []
    for _, name in stego_caps:
        if results[name]:
            # Replace infinity with a high value (100) for visualization
            finite_values = [min(v, 100.0) if not np.isinf(v) else 100.0 for v in results[name]]
            avg = np.mean(finite_values)
            avg_psnr.append(avg)
            labels.append(name)
    
    if avg_psnr:
        plt.figure(figsize=(10, 6))
        plt.bar(labels, avg_psnr)
        plt.ylabel('PSNR (dB)')
        plt.title('Average PSNR by Format')
        
        # Set reasonable y-limits
        max_finite_psnr = max(avg_psnr)
        min_finite_psnr = min(v for v in avg_psnr if not np.isinf(v))
        plt.ylim(max(0, min_finite_psnr - 5), max_finite_psnr + 5)
        
        # Add values on top of bars
        for i, v in enumerate(avg_psnr):
            if v >= 100:
                plt.text(i, v - 10, "Perfect Match", ha='center')
            else:
                plt.text(i, v + 0.5, f"{v:.2f} dB", ha='center')
        
        plt.tight_layout()
        plt.savefig('comparison/psnr_comparison.png')
        print("Saved PSNR comparison chart to comparison/psnr_comparison.png")
    
    # Release all captures
    orig_cap.release()
    for cap, _ in stego_caps:
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare steganographic videos across different formats")
    parser.add_argument("original", help="Path to original video")
    parser.add_argument("videos", nargs="+", help="Paths to steganographic video files to compare")
    
    args = parser.parse_args()
    
    # All videos including original
    all_videos = [args.original] + args.videos
    
    # Compare video properties and frames
    compare_videos(all_videos)
    
    # Calculate quality metrics
    calculate_video_quality(args.original, args.videos)
    
    print("\nDone!") 