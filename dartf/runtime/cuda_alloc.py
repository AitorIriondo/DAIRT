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

# ---- streams, events, pinned host memory, async copies (ctypes cudart; no torch) ----
_lib.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]; _lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]; _lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
_lib.cudaEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]; _lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]; _lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]; _lib.cudaStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
_lib.cudaMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]; _lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
_lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]; _lib.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
D2D = 3
class Stream:
    def __init__(self, blocking=False): p = ctypes.c_void_p(); _chk(_lib.cudaStreamCreateWithFlags(ctypes.byref(p), 0 if blocking else 1), "stream"); self.ptr = p.value   # 1 = cudaStreamNonBlocking; blocking streams order after legacy-stream cudaMemcpy
    def sync(self): _chk(_lib.cudaStreamSynchronize(self.ptr), "stream sync")
    def wait(self, ev): _chk(_lib.cudaStreamWaitEvent(self.ptr, ev.ptr, 0), "wait event")
class Event:
    def __init__(self): p = ctypes.c_void_p(); _chk(_lib.cudaEventCreateWithFlags(ctypes.byref(p), 2), "event"); self.ptr = p.value   # 2 = cudaEventDisableTiming
    def record(self, stream): _chk(_lib.cudaEventRecord(self.ptr, stream.ptr), "record")
    def sync(self): _chk(_lib.cudaEventSynchronize(self.ptr), "event sync")
class PinnedBuf:
    """page-locked host array (numpy view) for asynchronous uploads"""
    def __init__(self, shape, dtype):
        self.nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize; p = ctypes.c_void_p(); _chk(_lib.cudaMallocHost(ctypes.byref(p), max(self.nbytes, 1)), "cudaMallocHost"); self.ptr = p.value
        self.arr = np.ctypeslib.as_array((ctypes.c_uint8 * self.nbytes).from_address(self.ptr)).view(dtype).reshape(shape)
    def __del__(self):
        try: _lib.cudaFreeHost(self.ptr)
        except Exception: pass
def memcpy_async(dst_ptr, src_ptr, nbytes, kind, stream): _chk(_lib.cudaMemcpyAsync(dst_ptr, src_ptr, nbytes, kind, stream.ptr), "memcpyAsync")
def memcpy_d2d(dst_ptr, src_ptr, nbytes): _chk(_lib.cudaMemcpy(dst_ptr, src_ptr, nbytes, D2D), "D2D")
def memset_async(ptr, nbytes, stream): _chk(_lib.cudaMemsetAsync(ptr, 0, nbytes, stream.ptr), "memsetAsync")

# ---- CUDA graphs: capture the work a callable issues on a stream, replay it with one launch (removes per-kernel launch cost, which dominates small engines on the Orin) ----
_lib.cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]; _lib.cudaStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_lib.cudaGraphInstantiate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_ulonglong]; _lib.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.cudaGraphDestroy.argtypes = [ctypes.c_void_p]; _lib.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
class CudaGraph:
    def __init__(self, stream, fn):
        _chk(_lib.cudaStreamBeginCapture(stream.ptr, 2), "beginCapture")      # 2 = cudaStreamCaptureModeRelaxed
        try: fn()
        finally: g = ctypes.c_void_p(); rc = _lib.cudaStreamEndCapture(stream.ptr, ctypes.byref(g))
        _chk(rc, "endCapture"); e = ctypes.c_void_p(); _chk(_lib.cudaGraphInstantiate(ctypes.byref(e), g, 0), "graphInstantiate"); _lib.cudaGraphDestroy(g); self.exec = e.value
    def launch(self, stream): _chk(_lib.cudaGraphLaunch(self.exec, stream.ptr), "graphLaunch")
    def __del__(self):
        try: _lib.cudaGraphExecDestroy(self.exec)
        except Exception: pass
