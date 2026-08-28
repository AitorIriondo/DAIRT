"""Minimal cudart device buffer via ctypes (no pycuda/cuda-python dependency)."""
import ctypes, numpy as np
_lib = ctypes.CDLL("libcudart.so")
_lib.cudaMalloc.argtypes=[ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_lib.cudaFree.argtypes=[ctypes.c_void_p]
_lib.cudaMemcpy.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_lib.cudaDeviceSynchronize.argtypes=[]
H2D, D2H = 1, 2
def _chk(r, what):
    if r != 0: raise RuntimeError(f"{what} failed: cudaError {r}")
class DevBuf:
    def __init__(self, nbytes):
        self.nbytes=max(nbytes,1); p=ctypes.c_void_p(); _chk(_lib.cudaMalloc(ctypes.byref(p), self.nbytes), "cudaMalloc"); self.ptr=p.value
    def upload(self, arr):
        assert arr.nbytes <= self.nbytes; _chk(_lib.cudaMemcpy(self.ptr, arr.ctypes.data, arr.nbytes, H2D), "H2D")
    def download(self, shape, dt):
        out=np.empty(shape, dtype=dt); _chk(_lib.cudaMemcpy(out.ctypes.data, self.ptr, out.nbytes, D2H), "D2H"); return out
    @staticmethod
    def sync(): _chk(_lib.cudaDeviceSynchronize(), "sync")
    def __del__(self):
        try: _lib.cudaFree(self.ptr)
        except Exception: pass
