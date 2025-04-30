# GPU vs CPU Performance Comparison

## Embedding Performance

| Metric                 |   CPU |   GPU | Speedup   |
|:-----------------------|------:|------:|:----------|
| Raw Execution Time (s) | 35.08 | 36.02 | 0.97x     |
| Average PSNR (dB)      | 59.67 | 59.66 | -0.01     |
| Output File Size (MB)  |  5.13 |  5.13 | 1.00x     |

## Extraction Performance

| Metric                 |   CPU |   GPU | Speedup   |
|:-----------------------|------:|------:|:----------|
| Raw Execution Time (s) |  2.21 |  3.34 | 0.66x     |
| Bit Error Rate (%)     |  0.87 |  0.87 | +0.00     |

## Performance Visualization

### Processing Time Comparison

![Processing Time Comparison](performance_tests/time_comparison.png)

### DCT Processing Time Percentage

![DCT Processing Time Percentage](performance_tests/dct_percentage.png)

### GPU Speedup Comparison

![GPU Speedup Comparison](performance_tests/speedup_comparison.png)

