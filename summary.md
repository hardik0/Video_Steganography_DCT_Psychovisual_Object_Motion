# Python Implementation of Video Steganography Based on DCT Psychovisual Properties
## I. Introduction
### A. Objective
This report provides a functional Python implementation for a video steganography technique. The objective is to embed a secret message within a video file by modifying coefficients in the Discrete Cosine Transform (DCT) domain, based conceptually on methods described in academic literature exploring psychovisual properties and motion analysis. The implementation adheres to specific user-defined requirements regarding algorithms, libraries, and operational parameters.  

### B. Core Concepts
The technique leverages several core signal processing and steganographic concepts:

* Discrete Cosine Transform (DCT): A mathematical transform widely used in image and video compression (like JPEG and MPEG) to convert spatial pixel data into frequency coefficients, concentrating most visual energy into a few low-frequency coefficients.
* Inverse DCT (IDCT): The reverse transform used to reconstruct pixel data from DCT coefficients.
* Zig-Zag Scanning: A method to order the 2D DCT coefficients of a block into a 1D sequence, generally arranging them from low frequency to high frequency.
* Psychovisual Modulation: Exploiting characteristics of the human visual system (HVS). The HVS is less sensitive to changes in certain frequency components (particularly mid-to-high frequencies) than others. This technique embeds data by modifying specific mid-frequency DCT coefficients, aiming for minimal perceptual distortion.  
* Algorithmic Logic: The implementation follows specific logic, derived from user instructions referencing Algorithms 2, 3, and 4, for calculating embedding thresholds, embedding bits by modifying coefficient pairs, and extracting bits by comparing coefficient pairs.

### C. Key User Requirements
The implementation specifically addresses the following requirements:

1. Language and Libraries: Python, utilizing opencv-python for video I/O, numpy for numerical operations, scipy.fftpack for DCT/IDCT, and pickle for storing embedding locations.
2. Coefficient Selection: Target specific DCT coefficients within each 8x8 block, corresponding to indices 4, 5, 6, 7, 10, and 11 in the 1D zig-zag scan order.
3. Embedding Capacity: Embed 3 bits of the secret message into the 6 selected coefficients of a chosen block.
4. Algorithm Implementation: Implement functions reflecting the logic of:
    * Algorithm 2: Calculating thresholds f and s based on coefficient magnitudes and a parameter T=20.
    * Algorithm 3: Embedding 3 bits based on comparisons and modifications of coefficient pairs, using thresholds f and s.
    * Algorithm 4: Extracting 3 bits based on comparisons of the corresponding coefficient pairs.
5. Coordinate Tracking: Record the frame number and block coordinates (x, y) for every block where data is embedded and save this information to a separate file.
6. Separate Scripts: Provide distinct scripts for the embedding and extraction processes.
7. Simplified Motion Check: Implement a placeholder condition for selecting blocks, rather than complex motion vector analysis.

### D. Critical Simplification & Limitation: Motion Check
A crucial aspect of this implementation is the handling of block selection. The reference methodology likely utilizes motion vectors (MVs), typically available in compressed video streams (P-frames and B-frames), to identify blocks associated with object motion for data embedding. Accessing and interpreting codec-specific MVs reliably through standard libraries like OpenCV is complex and often not directly supported for arbitrary video files.  

Therefore, as per user instruction, this implementation replaces the motion vector analysis with a simplified placeholder condition for selecting blocks. Examples include embedding data in every Nth block or blocks within specific frame regions.

This simplification has significant implications. The original paper's rationale for using MVs (e.g., MV magnitude <= 7 ) likely aimed to embed data in regions with specific motion characteristics, potentially balancing imperceptibility and robustness against compression artifacts introduced during motion compensation. By decoupling block selection from the actual video motion content, this implementation will embed data in different locations than the original method would have. The chosen blocks might be static (where changes could be more perceptible due to temporal redundancy) or dynamic in ways not considered by the original algorithm. Consequently, the steganographic performance characteristics (imperceptibility, capacity distribution, robustness to potential subsequent compression) observed in the original research  may not directly apply to this simplified implementation. This deviation is a pragmatic choice for feasibility but represents a fundamental difference from the described academic method.  

### E. Irrelevance of Phishing Research
It is important to note that a significant portion of the background material reviewed  pertains to the domain of phishing detection using machine learning. This material covers datasets, features, algorithms, and implementations related to identifying malicious URLs or emails. This body of research is not relevant to the current task of implementing video steganography based on DCT modification. The implementation presented here relies solely on the user's specific instructions and the conceptual framework related to DCT-based steganography, as partially validated by snippet.  

