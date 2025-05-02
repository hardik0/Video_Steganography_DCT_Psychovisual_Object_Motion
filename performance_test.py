#!/usr/bin/env python3
import os
import time
import argparse
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def run_command(cmd, description="Running command"):
    """Run a shell command and return stdout"""
    print(f"{description}...")
    print(f"Command: {cmd}")
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    end_time = time.time()
    duration = end_time - start_time
    
    stdout_str = stdout.decode('utf-8')
    stderr_str = stderr.decode('utf-8')
    
    # Print any errors for debugging
    if stderr_str.strip():
        print(f"Error output: {stderr_str}")
    
    return stdout_str, stderr_str, duration

def parse_performance_metrics(output, operation_type):
    """Parse performance metrics from the output"""
    metrics = {}
    
    # Default values in case parsing fails
    metrics['total_time'] = 0
    metrics['dct_time'] = 0
    metrics['op_time'] = 0
    metrics['fps'] = 0
    
    if operation_type == "embed":
        metrics['motion_time'] = 0
        metrics['psnr'] = 0
    
    if operation_type == "extract":
        metrics['bit_error_rate'] = 0
    
    # Manually parse interesting lines
    for line in output.split('\n'):
        line = line.strip()
        
        if "Total processing time:" in line:
            try:
                metrics['total_time'] = float(line.split(":")[1].strip().split()[0])
            except:
                print(f"Failed to parse total time from: {line}")
        
        elif "DCT/IDCT operations:" in line or "DCT operations:" in line:
            try:
                metrics['dct_time'] = float(line.split(":")[1].strip().split()[0])
            except:
                print(f"Failed to parse DCT time from: {line}")
        
        elif "Embedding operations:" in line and operation_type == "embed":
            try:
                metrics['op_time'] = float(line.split(":")[1].strip().split()[0])
            except:
                print(f"Failed to parse embedding time from: {line}")
        
        elif "Extraction operations:" in line and operation_type == "extract":
            try:
                metrics['op_time'] = float(line.split(":")[1].strip().split()[0])
            except:
                print(f"Failed to parse extraction time from: {line}")
        
        elif "Motion detection:" in line and operation_type == "embed":
            try:
                metrics['motion_time'] = float(line.split(":")[1].strip().split()[0])
            except:
                print(f"Failed to parse motion time from: {line}")
        
        elif "Processing speed:" in line:
            try:
                metrics['fps'] = float(line.split(":")[1].strip().split()[0])
            except:
                print(f"Failed to parse fps from: {line}")
        
        elif "Average PSNR:" in line and operation_type == "embed":
            try:
                metrics['psnr'] = float(line.split(":")[1].strip().split()[0])
            except:
                print(f"Failed to parse PSNR from: {line}")
        
        elif "Bits differ in" in line and operation_type == "extract":
            try:
                parts = line.split("(")
                if len(parts) > 1:
                    metrics['bit_error_rate'] = float(parts[1].split("%")[0])
            except:
                print(f"Failed to parse bit error rate from: {line}")
    
    return metrics

def run_tests(input_video, message, format_preset="default"):
    """Run performance tests comparing CPU and GPU implementations"""
    results = {
        'cpu_embed': {},
        'gpu_embed': {},
        'cpu_extract': {},
        'gpu_extract': {}
    }
    
    # Test message to embed
    if not message:
        message = "This is a test message for steganography performance testing. It contains enough data to be meaningful for a performance comparison between CPU and GPU implementations."
    
    print(f"Testing with message: {message}")
    print(f"Message length: {len(message)} characters")
    
    # Create test directory
    os.makedirs("performance_tests", exist_ok=True)
    
    # Test embedding with CPU
    print("\n=== Testing embedding with CPU ===")
    print("Embedding with CPU...")
    cpu_embed_cmd = f"python3 embed.py {input_video} '{message}' performance_tests/cpu_stego.mp4 performance_tests/cpu_locations.pkl --format {format_preset}"
    print(f"Command: {cpu_embed_cmd}")
    cpu_embed_start = time.time()
    cpu_embed_output = subprocess.run(cpu_embed_cmd, shell=True, capture_output=True, text=True)
    cpu_embed_time = time.time() - cpu_embed_start
    if cpu_embed_output.returncode != 0:
        print(f"Error output: {cpu_embed_output.stderr}")
        return
    results['cpu_embed'] = parse_performance_metrics(cpu_embed_output.stdout, "embed")
    results['cpu_embed']['raw_time'] = cpu_embed_time
    
    # Test embedding with GPU
    print("\n=== Testing embedding with GPU ===")
    print("Embedding with GPU...")
    gpu_embed_cmd = f"python3 embed_gpu.py {input_video} '{message}' performance_tests/gpu_stego.mp4 performance_tests/gpu_locations.pkl --format {format_preset}"
    print(f"Command: {gpu_embed_cmd}")
    gpu_embed_start = time.time()
    gpu_embed_output = subprocess.run(gpu_embed_cmd, shell=True, capture_output=True, text=True)
    gpu_embed_time = time.time() - gpu_embed_start
    if gpu_embed_output.returncode != 0:
        print(f"Error output: {gpu_embed_output.stderr}")
        return
    results['gpu_embed'] = parse_performance_metrics(gpu_embed_output.stdout, "embed")
    results['gpu_embed']['raw_time'] = gpu_embed_time
    
    # Test extraction with CPU
    print("\n=== Testing extraction with CPU ===")
    print("Extracting with CPU...")
    cpu_extract_cmd = f"python3 extract.py performance_tests/cpu_stego.mp4 performance_tests/cpu_locations.pkl"
    print(f"Command: {cpu_extract_cmd}")
    cpu_extract_start = time.time()
    cpu_extract_output = subprocess.run(cpu_extract_cmd, shell=True, capture_output=True, text=True)
    cpu_extract_time = time.time() - cpu_extract_start
    if cpu_extract_output.returncode != 0:
        print(f"Error output: {cpu_extract_output.stderr}")
        return
    results['cpu_extract'] = parse_performance_metrics(cpu_extract_output.stdout, "extract")
    results['cpu_extract']['raw_time'] = cpu_extract_time
    
    # Test extraction with GPU
    print("\n=== Testing extraction with GPU ===")
    print("Extracting with GPU...")
    gpu_extract_cmd = f"python3 extract_gpu.py performance_tests/gpu_stego.mp4 performance_tests/gpu_locations.pkl"
    print(f"Command: {gpu_extract_cmd}")
    gpu_extract_start = time.time()
    gpu_extract_output = subprocess.run(gpu_extract_cmd, shell=True, capture_output=True, text=True)
    gpu_extract_time = time.time() - gpu_extract_start
    if gpu_extract_output.returncode != 0:
        print(f"Error output: {gpu_extract_output.stderr}")
        return
    results['gpu_extract'] = parse_performance_metrics(gpu_extract_output.stdout, "extract")
    results['gpu_extract']['raw_time'] = gpu_extract_time
    
    # Check video sizes
    if os.path.exists("performance_tests/cpu_stego.mp4"):
        results['cpu_embed']['file_size'] = os.path.getsize("performance_tests/cpu_stego.mp4") / (1024 * 1024)  # MB
    else:
        results['cpu_embed']['file_size'] = 0
        
    if os.path.exists("performance_tests/gpu_stego.mp4"):
        results['gpu_embed']['file_size'] = os.path.getsize("performance_tests/gpu_stego.mp4") / (1024 * 1024)  # MB
    else:
        results['gpu_embed']['file_size'] = 0
    
    # Calculate speedup (use raw time if metrics weren't properly extracted)
    if results['cpu_embed'].get('total_time', 0) > 0 and results['gpu_embed'].get('total_time', 0) > 0:
        results['embed_speedup'] = results['cpu_embed']['total_time'] / results['gpu_embed']['total_time']
    else:
        results['embed_speedup'] = results['cpu_embed']['raw_time'] / results['gpu_embed']['raw_time'] if results['gpu_embed']['raw_time'] > 0 else 1
    
    if results['cpu_extract'].get('total_time', 0) > 0 and results['gpu_extract'].get('total_time', 0) > 0:
        results['extract_speedup'] = results['cpu_extract']['total_time'] / results['gpu_extract']['total_time']
    else:
        results['extract_speedup'] = results['cpu_extract']['raw_time'] / results['gpu_extract']['raw_time'] if results['gpu_extract']['raw_time'] > 0 else 1
    
    return results

def generate_report(results, output_file="performance_report.md"):
    """Generate a markdown report with tables and graphs"""
    with open(output_file, "w") as f:
        f.write("# GPU vs CPU Performance Comparison\n\n")
        
        # Embedding Performance Summary
        f.write("## Embedding Performance\n\n")
        
        embed_table = []
        headers = ["Metric", "CPU", "GPU", "Speedup"]
        
        # Always include raw time as a fallback
        embed_table.append(["Raw Execution Time (s)", 
                           f"{results['cpu_embed'].get('raw_time', 0):.2f}", 
                           f"{results['gpu_embed'].get('raw_time', 0):.2f}", 
                           f"{results['embed_speedup']:.2f}x"])
        
        if results['cpu_embed'].get('total_time', 0) > 0 and results['gpu_embed'].get('total_time', 0) > 0:
            embed_table.append(["Total Processing Time (s)", 
                               f"{results['cpu_embed']['total_time']:.2f}", 
                               f"{results['gpu_embed']['total_time']:.2f}", 
                               f"{results['embed_speedup']:.2f}x"])
        
        if results['cpu_embed'].get('dct_time', 0) > 0 and results['gpu_embed'].get('dct_time', 0) > 0:
            cpu_dct_percent = results['cpu_embed']['dct_time'] / max(results['cpu_embed'].get('total_time', 1), 1) * 100
            gpu_dct_percent = results['gpu_embed']['dct_time'] / max(results['gpu_embed'].get('total_time', 1), 1) * 100
            dct_speedup = results['cpu_embed']['dct_time'] / max(results['gpu_embed']['dct_time'], 0.001)
            embed_table.append(["DCT Operations (s)", 
                               f"{results['cpu_embed']['dct_time']:.2f} ({cpu_dct_percent:.1f}%)", 
                               f"{results['gpu_embed']['dct_time']:.2f} ({gpu_dct_percent:.1f}%)", 
                               f"{dct_speedup:.2f}x"])
        
        if results['cpu_embed'].get('op_time', 0) > 0 and results['gpu_embed'].get('op_time', 0) > 0:
            cpu_op_percent = results['cpu_embed']['op_time'] / max(results['cpu_embed'].get('total_time', 1), 1) * 100
            gpu_op_percent = results['gpu_embed']['op_time'] / max(results['gpu_embed'].get('total_time', 1), 1) * 100
            op_speedup = results['cpu_embed']['op_time'] / max(results['gpu_embed']['op_time'], 0.001)
            embed_table.append(["Embedding Operations (s)", 
                               f"{results['cpu_embed']['op_time']:.2f} ({cpu_op_percent:.1f}%)", 
                               f"{results['gpu_embed']['op_time']:.2f} ({gpu_op_percent:.1f}%)", 
                               f"{op_speedup:.2f}x"])
        
        if results['cpu_embed'].get('motion_time', 0) > 0 and results['gpu_embed'].get('motion_time', 0) > 0:
            cpu_motion_percent = results['cpu_embed']['motion_time'] / max(results['cpu_embed'].get('total_time', 1), 1) * 100
            gpu_motion_percent = results['gpu_embed']['motion_time'] / max(results['gpu_embed'].get('total_time', 1), 1) * 100
            motion_speedup = results['cpu_embed']['motion_time'] / max(results['gpu_embed']['motion_time'], 0.001)
            embed_table.append(["Motion Detection (s)", 
                               f"{results['cpu_embed']['motion_time']:.2f} ({cpu_motion_percent:.1f}%)", 
                               f"{results['gpu_embed']['motion_time']:.2f} ({gpu_motion_percent:.1f}%)", 
                               f"{motion_speedup:.2f}x"])
        
        if results['cpu_embed'].get('fps', 0) > 0 and results['gpu_embed'].get('fps', 0) > 0:
            fps_improvement = results['gpu_embed']['fps'] / max(results['cpu_embed']['fps'], 0.001)
            embed_table.append(["Processing Speed (fps)", 
                               f"{results['cpu_embed']['fps']:.2f}", 
                               f"{results['gpu_embed']['fps']:.2f}", 
                               f"{fps_improvement:.2f}x"])
        
        if results['cpu_embed'].get('psnr', 0) > 0 and results['gpu_embed'].get('psnr', 0) > 0:
            psnr_diff = results['gpu_embed']['psnr'] - results['cpu_embed']['psnr']
            embed_table.append(["Average PSNR (dB)", 
                               f"{results['cpu_embed']['psnr']:.2f}", 
                               f"{results['gpu_embed']['psnr']:.2f}", 
                               f"{psnr_diff:+.2f}"])
        
        if results['cpu_embed'].get('file_size', 0) > 0 and results['gpu_embed'].get('file_size', 0) > 0:
            size_ratio = results['gpu_embed']['file_size'] / max(results['cpu_embed']['file_size'], 0.001)
            embed_table.append(["Output File Size (MB)", 
                               f"{results['cpu_embed']['file_size']:.2f}", 
                               f"{results['gpu_embed']['file_size']:.2f}", 
                               f"{size_ratio:.2f}x"])
        
        f.write(tabulate(embed_table, headers, tablefmt="pipe"))
        f.write("\n\n")
        
        # Extraction Performance Summary
        f.write("## Extraction Performance\n\n")
        
        extract_table = []
        
        # Always include raw time as a fallback
        extract_table.append(["Raw Execution Time (s)", 
                             f"{results['cpu_extract'].get('raw_time', 0):.2f}", 
                             f"{results['gpu_extract'].get('raw_time', 0):.2f}", 
                             f"{results['extract_speedup']:.2f}x"])
        
        if results['cpu_extract'].get('total_time', 0) > 0 and results['gpu_extract'].get('total_time', 0) > 0:
            extract_table.append(["Total Processing Time (s)", 
                                 f"{results['cpu_extract']['total_time']:.2f}", 
                                 f"{results['gpu_extract']['total_time']:.2f}", 
                                 f"{results['extract_speedup']:.2f}x"])
        
        if results['cpu_extract'].get('dct_time', 0) > 0 and results['gpu_extract'].get('dct_time', 0) > 0:
            cpu_dct_percent = results['cpu_extract']['dct_time'] / max(results['cpu_extract'].get('total_time', 1), 1) * 100
            gpu_dct_percent = results['gpu_extract']['dct_time'] / max(results['gpu_extract'].get('total_time', 1), 1) * 100
            dct_speedup = results['cpu_extract']['dct_time'] / max(results['gpu_extract']['dct_time'], 0.001)
            extract_table.append(["DCT Operations (s)", 
                                 f"{results['cpu_extract']['dct_time']:.2f} ({cpu_dct_percent:.1f}%)", 
                                 f"{results['gpu_extract']['dct_time']:.2f} ({gpu_dct_percent:.1f}%)", 
                                 f"{dct_speedup:.2f}x"])
        
        if results['cpu_extract'].get('op_time', 0) > 0 and results['gpu_extract'].get('op_time', 0) > 0:
            cpu_op_percent = results['cpu_extract']['op_time'] / max(results['cpu_extract'].get('total_time', 1), 1) * 100
            gpu_op_percent = results['gpu_extract']['op_time'] / max(results['gpu_extract'].get('total_time', 1), 1) * 100
            op_speedup = results['cpu_extract']['op_time'] / max(results['gpu_extract']['op_time'], 0.001)
            extract_table.append(["Extraction Operations (s)", 
                                 f"{results['cpu_extract']['op_time']:.2f} ({cpu_op_percent:.1f}%)", 
                                 f"{results['gpu_extract']['op_time']:.2f} ({gpu_op_percent:.1f}%)", 
                                 f"{op_speedup:.2f}x"])
        
        if results['cpu_extract'].get('fps', 0) > 0 and results['gpu_extract'].get('fps', 0) > 0:
            fps_improvement = results['gpu_extract']['fps'] / max(results['cpu_extract']['fps'], 0.001)
            extract_table.append(["Processing Speed (fps)", 
                                 f"{results['cpu_extract']['fps']:.2f}", 
                                 f"{results['gpu_extract']['fps']:.2f}", 
                                 f"{fps_improvement:.2f}x"])
        
        if results['cpu_extract'].get('bit_error_rate', 0) > 0 and results['gpu_extract'].get('bit_error_rate', 0) > 0:
            ber_diff = results['gpu_extract']['bit_error_rate'] - results['cpu_extract']['bit_error_rate']
            extract_table.append(["Bit Error Rate (%)", 
                                 f"{results['cpu_extract']['bit_error_rate']:.2f}", 
                                 f"{results['gpu_extract']['bit_error_rate']:.2f}", 
                                 f"{ber_diff:+.2f}"])
        
        f.write(tabulate(extract_table, headers, tablefmt="pipe"))
        f.write("\n\n")
        
        # Create graphs
        f.write("## Performance Visualization\n\n")
        
        # Total Processing Time Comparison
        plt.figure(figsize=(12, 6))
        
        # Use raw time as fallback if total_time is not available
        cpu_times = [
            results['cpu_embed'].get('total_time', results['cpu_embed'].get('raw_time', 0)),
            results['cpu_extract'].get('total_time', results['cpu_extract'].get('raw_time', 0))
        ]
        
        gpu_times = [
            results['gpu_embed'].get('total_time', results['gpu_embed'].get('raw_time', 0)),
            results['gpu_extract'].get('total_time', results['gpu_extract'].get('raw_time', 0))
        ]
        
        operations = ['Embedding', 'Extraction']
        x = np.arange(len(operations))
        width = 0.35
        
        plt.bar(x - width/2, cpu_times, width, label='CPU')
        plt.bar(x + width/2, gpu_times, width, label='GPU')
        
        plt.ylabel('Time (seconds)')
        plt.title('Processing Time Comparison')
        plt.xticks(x, operations)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('performance_tests/time_comparison.png')
        plt.close()
        
        f.write("### Processing Time Comparison\n\n")
        f.write("![Processing Time Comparison](performance_tests/time_comparison.png)\n\n")
        
        # DCT Time Percentage Comparison
        if all(key in results.get('cpu_embed', {}) and key in results.get('gpu_embed', {}) and 
               key in results.get('cpu_extract', {}) and key in results.get('gpu_extract', {}) 
               for key in ['dct_time', 'total_time']):
            
            plt.figure(figsize=(12, 6))
            
            cpu_dct_percent = [
                results['cpu_embed']['dct_time'] / max(results['cpu_embed']['total_time'], 0.001) * 100,
                results['cpu_extract']['dct_time'] / max(results['cpu_extract']['total_time'], 0.001) * 100
            ]
            
            gpu_dct_percent = [
                results['gpu_embed']['dct_time'] / max(results['gpu_embed']['total_time'], 0.001) * 100,
                results['gpu_extract']['dct_time'] / max(results['gpu_extract']['total_time'], 0.001) * 100
            ]
            
            plt.bar(x - width/2, cpu_dct_percent, width, label='CPU')
            plt.bar(x + width/2, gpu_dct_percent, width, label='GPU')
            
            plt.ylabel('Percentage of Total Time (%)')
            plt.title('DCT Processing Time Percentage')
            plt.xticks(x, operations)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('performance_tests/dct_percentage.png')
            plt.close()
            
            f.write("### DCT Processing Time Percentage\n\n")
            f.write("![DCT Processing Time Percentage](performance_tests/dct_percentage.png)\n\n")
        
        # Speedup Factor Comparison
        if 'embed_speedup' in results and 'extract_speedup' in results:
            plt.figure(figsize=(8, 6))
            
            speedups = [results['embed_speedup'], results['extract_speedup']]
            
            plt.bar(operations, speedups, color='green')
            plt.axhline(y=1.0, color='r', linestyle='-', label='No Speedup')
            
            plt.ylabel('Speedup Factor (CPU time / GPU time)')
            plt.title('GPU Speedup Comparison')
            
            plt.tight_layout()
            plt.savefig('performance_tests/speedup_comparison.png')
            plt.close()
            
            f.write("### GPU Speedup Comparison\n\n")
            f.write("![GPU Speedup Comparison](performance_tests/speedup_comparison.png)\n\n")
        
        print(f"Performance report generated: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare performance between CPU and GPU implementations')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('--message', help='Message to embed (default is a test message)')
    parser.add_argument('--format', choices=['default', 'lossless', 'raw', 'high_quality'], 
                        default='default', help='Output format preset (default: default)')
    
    args = parser.parse_args()
    
    results = run_tests(args.input_video, args.message, args.format)
    generate_report(results) 