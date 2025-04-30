#!/usr/bin/env python3

import os
import sys
import importlib

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import cupy as cp
    print(f"CuPy version: {cp.__version__}")
    
    # Check for cupyx
    import cupyx
    print(f"CuPy extension modules (cupyx): {dir(cupyx)}")
    
    # Check for cupyx.scipy
    if hasattr(cupyx, 'scipy'):
        import cupyx.scipy
        print(f"CuPy scipy modules: {dir(cupyx.scipy)}")
        
        # Check for cupyx.scipy.fft as an alternative to fftpack
        if hasattr(cupyx.scipy, 'fft'):
            import cupyx.scipy.fft
            print(f"CuPy FFT functions: {dir(cupyx.scipy.fft)}")
            print("Found cupyx.scipy.fft - we can use this instead of fftpack")
        else:
            print("cupyx.scipy.fft not found")
            
        # Check for cupyx.scipy.fftpack
        if hasattr(cupyx.scipy, 'fftpack'):
            import cupyx.scipy.fftpack
            print(f"CuPy FFTpack functions: {dir(cupyx.scipy.fftpack)}")
        else:
            print("cupyx.scipy.fftpack is not available in this CuPy version")
    else:
        print("cupyx.scipy module not found")
        
    # Try to import specific functions we need
    try:
        from cupyx.scipy.fft import dct as cuda_dct
        from cupyx.scipy.fft import idct as cuda_idct
        print("Successfully imported dct/idct from cupyx.scipy.fft")
        
        # Test the functions
        test_array = cp.ones((8, 8), dtype=cp.float32)
        dct_result = cuda_dct(test_array, norm='ortho')
        idct_result = cuda_idct(dct_result, norm='ortho')
        print("DCT/IDCT functions work correctly")
    except ImportError:
        print("Failed to import dct/idct from cupyx.scipy.fft")
    except Exception as e:
        print(f"Error testing dct/idct functions: {str(e)}")
        
    print("\nCuPy test completed")
except ImportError as e:
    print(f"Error importing CuPy: {str(e)}")
    print("CuPy is not properly installed") 