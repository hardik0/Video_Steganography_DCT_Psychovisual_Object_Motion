# Video Steganography Format Test Results

This document contains test results for different video formats used with our DCT-based steganography system. We tested various formats to determine the optimal balance between file size, visual quality, and steganographic data integrity.

## Test Setup

- **Base Video**: SampleVideo_1280x720_2mb.mp4 (1280x720 resolution, 25fps, 337 frames)
- **Test Messages**:
  - Simple message: "Testing lossless format" (for lossless AVI)
  - Medium message: "Testing high quality format" (for high quality MJPG)
  - Original test: "Video steganography based on DCT psychovisual and object motion" (for MP4)
- **Error Correction**: 5x repetition code (each bit repeated 5 times)
- **Embedding Method**: DCT coefficient modification with motion detection

## Test Results

### Summary

| Format | Codec | File Size | Bit Error Rate | PSNR | Normalized Correlation |
|--------|-------|-----------|----------------|------|------------------------|
| Lossless AVI | HFYU | 347 MB | 4.35% | 76.62 dB | 0.9375 |
| High Quality AVI | MJPG | 31 MB | 3.24% | 41.31 dB | 0.9515 |
| MP4 | mp4v | 5.2 MB | 1.79% | 38.56 dB | 0.9749 |

### Detailed Results

#### 1. Lossless Format (Huffman YUV)

**Command:**
```
python embed.py SampleVideo_1280x720_2mb.mp4 "Testing lossless format" stego_lossless.mp4 locations.pkl --format lossless
```

**Key Results:**
- File Size: 347 MB (165x larger than original)
- Embedding Capacity: 184 bits (23 characters without error correction)
- Total Embedded Data: 920 bits (with 5x repetition)
- Quality Metrics:
  - Average PSNR: 76.62 dB
  - Minimum PSNR: 59.29 dB
  - Maximum PSNR: Perfect match in later frames
- Extraction Results:
  - Original Message: "Testing lossless format"
  - Extracted Message: "Tf²4Ing lossless form`t"
  - Bit Error Rate: 4.35% (8 errors out of 184 bits)
  - Normalized Correlation: 0.9375

**Observations:**
- Extremely large file size despite being a short video
- Most errors occurred in the first few bytes (character errors)
- Some words remained intact ("lossless form")
- PSNR values indicate excellent visual quality
- Higher compression resistance might be needed even with lossless codec

#### 2. High Quality Format (Motion JPEG)

**Command:**
```
python embed.py SampleVideo_1280x720_2mb.mp4 "Testing high quality format" stego_hq.mp4 locations.pkl --format high_quality
```

**Key Results:**
- File Size: 31 MB (15x larger than original)
- Embedding Capacity: 216 bits (27 characters without error correction)
- Total Embedded Data: 1080 bits (with 5x repetition)
- Quality Metrics:
  - Average PSNR: 41.31 dB
  - Minimum PSNR: 38.77 dB
  - Maximum PSNR: 42.17 dB
- Extraction Results:
  - Original Message: "Testing high quality format"
  - Extracted Message: "Tf²4Ing high quality format"
  - Bit Error Rate: 3.24% (7 errors out of 216 bits)
  - Normalized Correlation: 0.9515

**Observations:**
- More reasonable file size than lossless format
- Errors concentrated in first few characters, while "high quality format" remained intact
- Better bit error rate than lossless format
- Good balance between file size and data integrity
- Very similar PSNR values to lossless, suggesting similar visual quality

#### 3. Default Format (MP4)

**Command:**
```
python embed.py SampleVideo_1280x720_2mb.mp4 "Video steganography based on DCT psychovisual and object motion" steganographic_video.mp4 locations.pkl
```

**Key Results:**
- File Size: 5.2 MB (2.5x larger than original)
- Embedding Capacity: 504 bits (63 characters without error correction)
- Total Embedded Data: 2520 bits (with 5x repetition)
- Quality Metrics:
  - Average PSNR: 38.56 dB
  - Minimum PSNR: 36.62 dB
  - Maximum PSNR: 39.36 dB
- Extraction Results:
  - Original Message: "Video steganography based on DCT psychovisual and object motion"
  - Extracted Message: "Vj¤%K steganography b`sed on DCT psychovisual and object motioo"
  - Bit Error Rate: 1.79% (9 errors out of 504 bits)
  - Normalized Correlation: 0.9749

**Observations:**
- Smallest file size among tested formats
- Best bit error rate despite using a more lossy compression format
- Most of the message remained intact and readable
- Error patterns suggest specific DCT coefficients are being affected by compression
- Acceptable PSNR values, with some visible differences from original video

## Visual Comparison

Our automated comparison tool generated frame comparisons at different positions in the video. The results revealed:

1. **Frame Appearance**:
   - All formats maintain good visual quality with no obvious artifacts visible to casual viewers
   - The lossless format (HFYU) is virtually indistinguishable from the original
   - MJPG format shows very slight color changes in some areas
   - MP4 format has the most noticeable changes, but still maintains good quality

2. **PSNR Analysis**:
   - Lossless format: Extremely high PSNR values (60-100+ dB), with some frames being perfect matches
   - High Quality MJPG: Good PSNR around 41 dB
   - MP4: Acceptable PSNR around 38-39 dB

3. **Frame-by-Frame Behavior**:
   - All formats maintain consistent quality throughout the video
   - Early frames (0, 30) tend to have slightly lower PSNR than middle frames (60, 90)
   - Lossless format achieves perfect matches in later frames (90, 120)

4. **Perceived Quality vs. File Size**:
   - The visual quality increase from MP4 (5.2MB) to MJPG (31MB) is noticeable but modest
   - The jump from MJPG (31MB) to lossless HFYU (347MB) yields diminishing returns in perceived quality

## Findings and Recommendations

1. **Unexpected Result**: MP4 format (most compressed) had the lowest bit error rate, contrary to expectations. This may be because:
   - Our DCT-based approach aligns well with MP4's own DCT-based compression algorithm
   - The specific coefficient positions we modified may be less affected by MP4 compression
   - Higher thresholds used for MP4 embedding may have made the changes more robust

2. **File Size Considerations**:
   - Lossless format (HFYU): Produces prohibitively large files (347MB)
   - High Quality (MJPG): Reasonable compromise (31MB)
   - MP4: Most efficient (5.2MB) with best error rates

3. **Visual Quality**:
   - All formats maintained good visual quality
   - PSNR values were excellent for lossless (76.62 dB) and good for MJPG (41.31 dB) and MP4 (38.56 dB)
   - The human eye would struggle to identify differences in any of these formats

4. **Best Format Choice**:
   - For general use: MP4 format offers the best balance of size and accuracy
   - For maximum message integrity in moderate file size: High Quality MJPG
   - Lossless format is not recommended due to extreme file size without improved accuracy

## Future Optimization Suggestions

1. **Format-Specific Coefficient Selection**:
   - Analyze which DCT coefficients are least affected by each codec
   - Tailor coefficient selection to the specific format being used

2. **Adaptive Thresholds**:
   - Implement dynamic thresholds based on the codec being used
   - Further increase thresholds for more lossy formats

3. **Error Distribution Analysis**:
   - Study error patterns to identify which parts of message are most vulnerable
   - Implement selective redundancy for vulnerable data sections

4. **Advanced Error Correction**:
   - Replace simple repetition with more sophisticated error correction
   - Implement Reed-Solomon codes for better error recovery with less overhead 