# GPU vs CPU Performance Comparison

## Embedding Performance

| Metric                 |   CPU |   GPU | Speedup   |
|:-----------------------|------:|------:|:----------|
| Raw Execution Time (s) | 16.29 | 18.3  | 0.89x     |
| Average PSNR (dB)      | 58.7  | 58.7  | +0.00     |
| Output File Size (MB)  |  5.11 |  5.11 | 1.00x     |

## Extraction Performance

| Metric                 |   CPU |   GPU | Speedup   |
|:-----------------------|------:|------:|:----------|
| Raw Execution Time (s) |  2.31 |  3.14 | 0.74x     |
| Bit Error Rate (%)     |  2.56 |  2.56 | +0.00     |

## Performance Visualization

### Processing Time Comparison

![Processing Time Comparison](performance_tests/time_comparison.png)

### DCT Processing Time Percentage

![DCT Processing Time Percentage](performance_tests/dct_percentage.png)

### GPU Speedup Comparison

![GPU Speedup Comparison](performance_tests/speedup_comparison.png)

