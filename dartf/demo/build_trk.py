"""Build FP16 TensorRT plans for the tracker graphs with named dynamic dimensions.
usage: python build_trk.py graph.onnx out.plan "B=1:4:32,M=1:7:7,P=1:8:16" [--fp32]
Every input dimension of size -1 is matched by its position to the named dims in the order they appear in the ONNX (dim_param names are used when present)."""
import tensorrt as trt, sys, time, onnx
onnx_path, plan, spec = sys.argv[1], sys.argv[2], sys.argv[3]; fp32 = "--fp32" in sys.argv
if "--plugins" in sys.argv:
    import ctypes; lib = sys.argv[sys.argv.index("--plugins") + 1]; reg = trt.get_plugin_registry()
    try: reg.load_library(lib)
    except Exception as e: print("load_library:", e)
    if reg.get_creator("LQMemAttn", "1", "") is None: h = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL); h.lq_register(); print("registered via ctypes")
    print("LQMemAttn creator:", reg.get_creator("LQMemAttn", "1", "") is not None)
rng = {k: tuple(int(v) for v in r.split(":")) for k, r in (p.split("=") for p in spec.split(","))}
m = onnx.load(onnx_path, load_external_data=False); names = {}
for i in m.graph.input:
    dims = [(d.dim_param or None) if d.dim_value == 0 else d.dim_value for d in i.type.tensor_type.shape.dim]; names[i.name] = dims
logger = trt.Logger(trt.Logger.WARNING); b = trt.Builder(logger); net = b.create_network(0); p = trt.OnnxParser(net, logger)
assert p.parse_from_file(onnx_path), [p.get_error(i) for i in range(p.num_errors)]
cfg = b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
if not fp32: cfg.set_flag(trt.BuilderFlag.FP16)
if "--int8" in sys.argv: cfg.set_flag(trt.BuilderFlag.INT8)
prof = b.create_optimization_profile()
for i in range(net.num_inputs):
    t = net.get_input(i); s = list(t.shape)
    if -1 in s:
        mn, op, mx = list(s), list(s), list(s)
        for d, v in enumerate(s):
            if v == -1:
                nm = names[t.name][d]; assert nm in rng, f"no range for dim {nm} of {t.name}"; mn[d], op[d], mx[d] = rng[nm]
        prof.set_shape(t.name, mn, op, mx); print("profile", t.name, mn, op, mx)
cfg.add_optimization_profile(prof); t0 = time.time(); ser = b.build_serialized_network(net, cfg); assert ser is not None
open(plan, "wb").write(ser); import os; print("built", plan, "%.0fs" % (time.time() - t0), os.path.getsize(plan) // 1000000, "MB")
