import ctypes, sys, tensorrt as trt
logger = trt.Logger(trt.Logger.INFO); reg = trt.get_plugin_registry()
try: ok = reg.load_library(sys.argv[1]); print("load_library ->", ok)
except Exception as e: print("load_library failed:", e); h = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL); h.lq_register(); print("ctypes + lq_register ok")
c = reg.get_creator("LQFc1Gelu", "1", ""); print("get_creator:", c)
allc = reg.all_creators; print(len(allc), "creators; sample:", [(getattr(x, "name", None), getattr(x, "plugin_version", None)) for x in allc[:3]], "| LQ:", [(getattr(x, "name", None)) for x in allc if str(getattr(x, "name", "")).startswith("LQ")])
if c is None:
    h = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL); h.lq_register(); c = reg.get_creator("LQFc1Gelu", "1", ""); print("after explicit lq_register:", c)
