"""Build an FP16 TensorRT plan for the mask head with a dynamic prompt axis K (profile min/opt/max)."""
import tensorrt as trt, sys, time
onnx, plan, kmin, kopt, kmax = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
logger = trt.Logger(trt.Logger.WARNING); b = trt.Builder(logger); net = b.create_network(0); p = trt.OnnxParser(net, logger)
assert p.parse_from_file(onnx), [p.get_error(i) for i in range(p.num_errors)]
cfg = b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
if "--int8" in sys.argv: cfg.set_flag(trt.BuilderFlag.INT8)
prof = b.create_optimization_profile()
for i in range(net.num_inputs):
    t = net.get_input(i); s = list(t.shape)
    if -1 in s:
        d = s.index(-1); mn, op, mx = list(s), list(s), list(s); mn[d], op[d], mx[d] = kmin, kopt, kmax
        prof.set_shape(t.name, mn, op, mx); print("profile", t.name, mn, op, mx)
cfg.add_optimization_profile(prof); t0 = time.time(); ser = b.build_serialized_network(net, cfg); assert ser is not None
open(plan, "wb").write(ser); print("built", plan, "%.0fs" % (time.time() - t0))
