import numpy as np
from scipy.fftpack import dct, idct

def block_dct(block):
    """Apply 2D DCT to an 8x8 block"""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def block_idct(block):
    """Apply 2D IDCT to an 8x8 block"""
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

def select_blocks(frame, block_size=8):
    """Select blocks for embedding based on simplified motion check"""
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