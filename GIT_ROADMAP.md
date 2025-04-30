# Git Branching Strategy and Version Organization

## Branch Structure

1. **master** (V2 with GPU support)
   - Current stable version with GPU acceleration
   - Contains both CPU and GPU implementations
   - Main development branch

2. **v1** (CPU-only version)
   - Original CPU-only implementation
   - Stable version without GPU support
   - Based on v1_improvement.md specifications

3. **future** (Pure GPU version)
   - Optimized GPU-only implementation
   - Contains all GPU-related optimizations
   - Based on gpu_optimization_roadmap.md

## Version Details

### V1 (CPU-only)
- Original implementation
- Focus on CPU optimization
- Based on v1_improvement.md
- Key files:
  - `steganography.py`
  - `motion_detection.py`
  - `dct_operations.py`
  - `utils.py`

### V2 (Hybrid CPU/GPU)
- Current master branch
- Supports both CPU and GPU operations
- Key files:
  - `steganography.py`
  - `motion_detection.py`
  - `dct_operations.py`
  - `utils.py`
  - `gpu_operations.py`
  - `gpu_test.py`
  - `gpu_performance_summary.md`

### Future (GPU-optimized)
- Pure GPU implementation
- Optimized for GPU performance
- Key files:
  - `gpu_steganography.py`
  - `gpu_motion_detection.py`
  - `gpu_dct_operations.py`
  - `gpu_utils.py`
  - `gpu_optimization_roadmap.md`
  - `gpu_performance_summary.md`

## Branch Management

### Creating Branches
```bash
# Create and switch to v1 branch
git checkout -b v1

# Create and switch to future branch
git checkout -b future
```

### Merging Strategy
- Feature branches should be created from master
- Hotfixes can be created from any branch
- Regular merges from master to future
- No merges from future to master until stable

### Version Tags
- v1.0.0 - CPU-only stable version
- v2.0.0 - Current hybrid CPU/GPU version
- v3.0.0 - Future GPU-optimized version

## Development Workflow

1. **V1 Development**
   - Work on v1 branch
   - Focus on CPU optimizations
   - No GPU-related changes

2. **V2 Development**
   - Work on master branch
   - Maintain both CPU and GPU support
   - Regular performance testing

3. **Future Development**
   - Work on future branch
   - Focus on GPU optimizations
   - Follow gpu_optimization_roadmap.md

## Documentation

Each branch should maintain its own documentation:
- README.md - Branch-specific instructions
- CHANGELOG.md - Version-specific changes
- Performance reports
- Optimization roadmaps

## Testing Strategy

1. **V1 Testing**
   - CPU performance benchmarks
   - Steganography accuracy tests
   - Memory usage analysis

2. **V2 Testing**
   - Both CPU and GPU benchmarks
   - Cross-platform compatibility
   - Feature parity tests

3. **Future Testing**
   - GPU performance optimization
   - Memory transfer optimization
   - Batch processing tests 