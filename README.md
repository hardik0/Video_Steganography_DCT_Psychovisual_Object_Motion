# Video Steganography with DCT, Motion Analysis and Psychovisual Thresholds

This is an implementation of video steganography based on DCT (Discrete Cosine Transform) coefficients, object motion analysis, and psychovisual thresholds. The implementation is inspired by the research paper ["Video steganography based on DCT psychovisual and object motion"](https://beei.org/index.php/EEI/article/view/1859) by Muhammad Fuad and Ferda Ernawan.

## Features

- **DCT-based embedding**: Uses middle frequency DCT coefficients (4, 5, 6, 7, 10, 11) to hide data
- **Motion detection**: Embeds data in regions with motion to improve imperceptibility
- **Psychovisual thresholds**: Uses human visual perception models to determine optimal thresholds
- **Strong error correction**: Implements bit repetition (5x) for robust message extraction
- **Quality metrics**: Calculates PSNR (Peak Signal-to-Noise Ratio) to measure video quality
- **Multiple formats support**: Supports optimized embedding for various codecs and formats
- **MPEG-4 resistance**: Optimized for resilience against video compression
- **GPU acceleration**: Provides CUDA-accelerated DCT processing for improved performance

## Requirements

```
opencv-python
numpy
scipy
matplotlib
tabulate
cupy-cuda11x  # For GPU acceleration
```

Install the requirements using:
```
pip install opencv-python numpy scipy matplotlib tabulate
pip install cupy-cuda11x  # Only if CUDA is available
```

## Components

1. **dct_utils.py**: Utility functions for DCT operations, motion detection, and block selection
2. **embed.py**: Script for embedding a secret message into a video
3. **extract.py**: Script for extracting the hidden message from a steganographic video
4. **dct_utils_gpu.py**: GPU-accelerated utility functions for DCT operations
5. **embed_gpu.py**: GPU-accelerated script for embedding messages
6. **extract_gpu.py**: GPU-accelerated script for extracting hidden messages
7. **compare_formats.py**: Script for comparing different video format results
8. **FORMAT_TESTS.md**: Detailed analysis of our format comparison tests
9. **performance_test.py**: Script for comparing CPU vs GPU performance

## Usage

### Embedding a message

#### CPU Version:
```bash
python embed.py input_video.mp4 "Your secret message" output_video.mp4 locations.pkl [--repetition 5] [--format FORMAT]
```

#### GPU Version:
```bash
python embed_gpu.py input_video.mp4 "Your secret message" output_video.mp4 locations.pkl [--repetition 5] [--format FORMAT] [--no-gpu]
```

Arguments:
- `input_video.mp4`: Path to the input video file
- `"Your secret message"`: The secret message to embed
- `output_video.mp4`: Path to save the steganographic video
- `locations.pkl`: File to save embedding locations
- `--repetition`: Optional repetition factor for error correction (default: 5)
- `--format`: Output format preset (choices: default, lossless, raw, high_quality)
- `--no-gpu`: Disable GPU acceleration (for embed_gpu.py)

### Format Presets

- **default**: Standard MP4 format (most compatible but may lose data with compression)
- **lossless**: Uses lossless AVI codecs for best data preservation
- **raw**: Uses uncompressed YUV formats for maximum data integrity
- **high_quality**: Uses high-quality Motion JPEG for balance of quality and size

### Extracting a message

#### CPU Version:
```bash
python extract.py steganographic_video.mp4 locations.pkl [--repetition 5]
```

#### GPU Version:
```bash
python extract_gpu.py steganographic_video.mp4 locations.pkl [--repetition 5] [--no-gpu]
```

Arguments:
- `steganographic_video.mp4`: Path to the steganographic video
- `locations.pkl`: File containing embedding locations
- `--repetition`: Optional repetition factor for error correction (default: 5)
- `--no-gpu`: Disable GPU acceleration (for extract_gpu.py)

### Comparing formats

```bash
python compare_formats.py original.mp4 stego_file1.mp4 stego_file2.avi [other_stego_files...]
```

Arguments:
- `original.mp4`: Path to the original video file
- `stego_file1.mp4 stego_file2.avi...`: Paths to steganographic videos to compare

### Performance Testing

```bash
python performance_test.py input_video.mp4 [--message "Your test message"] [--format FORMAT]
```

Arguments:
- `input_video.mp4`: Path to the input video file
- `--message`: Optional test message to embed (default is a predefined test message)
- `--format`: Output format preset (choices: default, lossless, raw, high_quality)

The performance test will:
1. Run embedding and extraction with both CPU and GPU implementations
2. Measure execution times for different processing stages
3. Generate a performance report with tables and graphs
4. Create a `performance_report.md` file with the results

## GPU Acceleration

The GPU-accelerated implementation provides significant performance improvements, especially for larger videos:

- **Batch Processing**: Processes multiple DCT blocks in parallel
- **Cupy Integration**: Uses Cupy library for CUDA-accelerated operations
- **Automatic Fallback**: Falls back to CPU processing when CUDA is unavailable
- **Performance Metrics**: Provides detailed timing statistics for different processing stages

To use GPU acceleration, you need:
1. NVIDIA GPU with CUDA support
2. CUDA toolkit installed
3. Cupy library installed with appropriate CUDA version (e.g., `cupy-cuda11x`)

## Format Tests

We conducted extensive tests with different video formats to find the optimal balance between file size, visual quality, and steganographic robustness. Our findings include:

- **MP4** (default): Smallest files (5.2MB), best bit error rate (1.79%), PSNR: 38.56 dB
- **High Quality AVI** (MJPG): Moderate size (31MB), good bit error rate (3.24%), PSNR: 41.31 dB
- **Lossless AVI** (HFYU): Very large files (347MB), highest PSNR (76.62 dB), but higher bit error rate (4.35%)

For detailed test results and analysis, see [FORMAT_TESTS.md](FORMAT_TESTS.md).

## Sample Video

For testing, we used the "Big Buck Bunny" sample video (SampleVideo_1280x720_2mb.mp4) which is commonly used for video processing tests. The original video can be found at: [https://www.sample-videos.com/video321/mp4/720/big_buck_bunny_720p_2mb.mp4](https://www.sample-videos.com/video321/mp4/720/big_buck_bunny_720p_2mb.mp4)

## How It Works

1. **Motion Detection**: The algorithm analyzes frame differences to detect regions with motion.
2. **Block Selection**: 8x8 pixel blocks are selected preferentially in motion areas, as changes in these regions are less noticeable to human perception.
3. **DCT Transformation**: Each selected block is transformed using DCT.
4. **Coefficient Modification**: Specific middle frequency DCT coefficients are modified to embed bits from the secret message.
5. **Psychovisual Thresholds**: The modification amounts are determined based on human visual perception models.
6. **Error Correction**: Each bit is repeated multiple times for robustness against compression and noise.
7. **Quality Analysis**: The algorithm calculates PSNR to measure the quality impact of embedding.

## Output Analysis

The implementation generates several analysis files:

- `quality_analysis/`: Contains PSNR values and visual comparisons of original vs modified frames
- `extraction_analysis/`: Contains information about extracted bits and DCT coefficients
- `original_binary.txt` and `extracted_binary.txt`: For comparing original and extracted bit patterns
- `comparison/`: Frame-by-frame comparisons of different video formats (when using compare_formats.py)
- `performance_tests/`: Performance comparison data between CPU and GPU implementations

## Performance Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality difference between original and steganographic video
- **NC (Normalized Correlation)**: Measures the correlation between original and extracted message bits
- **Bit Error Rate**: Percentage of incorrectly extracted bits
- **Processing Speed**: Frames per second for embedding and extraction
- **DCT Processing Time**: Time spent on DCT/IDCT operations

## Format Comparison

| Format | Advantage | Disadvantage | Best Use Case |
|--------|-----------|--------------|--------------|
| MP4 (default) | Smallest file size, Most compatible | Higher data loss due to compression | For casual use where compatibility is prioritized |
| Lossless AVI | No data loss from compression | Large file size | For maximum message integrity |
| Raw YUV | Perfect message preservation | Very large file size | For critical data where size isn't a concern |
| High Quality MJPG | Good compromise of quality and size | Less compatible than MP4 | For general purpose steganography with good quality |

## CPU vs GPU Performance

Performance improvements with GPU acceleration depend on:
- Video size and resolution
- Amount of data being embedded
- CUDA device capabilities
- Processing stage (DCT operations benefit most)

Typical speedups observed:
- **DCT Operations**: 3-10x faster on GPU
- **Overall Embedding**: 2-5x faster on GPU
- **Extraction Process**: 2-4x faster on GPU

For detailed performance comparison, run the `performance_test.py` script.

## Limitations

- Works best with uncompressed or lightly compressed video formats
- Performance may vary depending on video content and motion characteristics
- Large messages require videos with sufficient frames and motion
- Resistance to attacks other than compression is not guaranteed
- GPU acceleration requires NVIDIA GPU with CUDA support

## Future Improvements

- Implement more advanced error correction codes (e.g., Reed-Solomon)
- Add support for more codecs and container formats
- Integrate with deep learning models for better motion detection
- Implement adaptive block selection based on content complexity
- Add support for AMD GPUs with ROCm

## References

- Original research paper: [Video Steganography Based on DCT Psychovisual Properties](https://beei.org/index.php/EEI/article/download/1859/2001) 