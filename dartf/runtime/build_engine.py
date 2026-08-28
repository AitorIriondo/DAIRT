"""Build a TensorRT plan from a (Q/DQ) ONNX. Flags: --int8 --fp16 (default both), --workspace-gb.
Writes <plan>.layers.json with per-layer precision from the EngineInspector and prints an INT8 tally."""
import tensorrt as trt, sys, json, time, argparse, collections
ap=argparse.ArgumentParser(); ap.add_argument("onnx"); ap.add_argument("plan")
ap.add_argument("--no-int8", action="store_true"); ap.add_argument("--no-fp16", action="store_true")
ap.add_argument("--workspace-gb", type=float, default=8); ap.add_argument("--timing-cache", default=None)
ap.add_argument("--verbose", action="store_true"); ap.add_argument("--obey", action="store_true", help="OBEY_PRECISION_CONSTRAINTS")
ap.add_argument("--fp32-layers", default=None, help="regex: layers whose name matches are pinned to FP32 (implies --obey)")
ap.add_argument("--strongly-typed", action="store_true")
ap.add_argument("--sparse", action="store_true", help="enable 2:4 structured-sparsity tactics (BuilderFlag.SPARSE_WEIGHTS)")
ap.add_argument("--dart-mode", default=None, choices=["attn-v-only","norm-only","norm-softmax-reduce","attn-core","attention","all"], help="DART build_engine.py mixed-precision mode (type-based FP32 pinning + OBEY)")
ap.add_argument("--plugins", default=None, help="comma list of plugin .so files to load (ctypes) before parsing")
ap.add_argument("--mark-outputs", default=None, help="regex: tensors whose name matches are marked as network outputs (fusion breaker)")
a=ap.parse_args()
logger=trt.Logger(trt.Logger.VERBOSE if a.verbose else trt.Logger.WARNING)
if a.plugins:
    import ctypes
    reg = trt.get_plugin_registry()
    for lib in a.plugins.split(","):
        try: reg.load_library(lib)
        except Exception as e: print("registry load_library failed:", e)
        if reg.get_creator("LQFc1Gelu", "1", "") is None:
            h = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL); h.lq_register(); print("registered LQ plugins via ctypes:", lib)
    print("creators:", [c.name for c in reg.all_creators if c.name.startswith("LQ")])
b=trt.Builder(logger); net=b.create_network(int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if a.strongly_typed else 0); p=trt.OnnxParser(net,logger)
t=time.time()
import os as _os
if _os.path.exists(a.onnx+".data") or _os.path.getsize(a.onnx) < 50_000_000:
    ok=p.parse_from_file(a.onnx)          # resolves external data relative to the file
else:
    with open(a.onnx,"rb") as f: ok=p.parse(f.read())
if not ok:
    for i in range(p.num_errors): print("PARSE ERROR:", p.get_error(i))
    sys.exit(1)
print("parsed %d layers in %.0fs"%(net.num_layers,time.time()-t))
if a.mark_outputs:
    import re as _re; rx=_re.compile(a.mark_outputs); n_mark=0
    for i in range(net.num_layers):
        L=net.get_layer(i)
        for j in range(L.num_outputs):
            t=L.get_output(j)
            if rx.search(t.name) and not t.is_network_output: net.mark_output(t); n_mark+=1
    print("marked %d tensors as outputs"%n_mark)
if a.dart_mode:
    # Port of DART sam3/trt/build_engine.py::_apply_mixed_precision (type-based selection).
    skip=set(getattr(trt.LayerType,t) for t in ("SHAPE","CONSTANT","IDENTITY","SHUFFLE","GATHER","SLICE","SQUEEZE","UNSQUEEZE","CONCATENATION","CONDITION","CAST","ASSERTION","FILL","SCATTER","RESIZE","NON_ZERO","ONE_HOT","GRID_SAMPLE","CONDITIONAL_INPUT","CONDITIONAL_OUTPUT") if hasattr(trt.LayerType,t))
    SM=getattr(trt.LayerType,"SOFTMAX",None); MM=getattr(trt.LayerType,"MATRIX_MULTIPLY",None); NM=getattr(trt.LayerType,"NORMALIZATION",None); RD=getattr(trt.LayerType,"REDUCE",None)
    n32=0; mode=a.dart_mode
    for i in range(net.num_layers):
        L=net.get_layer(i)
        if L.type in skip: continue
        f=False; name=L.name
        if mode=="attn-v-only": f = L.type in (SM,NM) or (L.type==MM and "/attn/MatMul_1" in name)
        elif mode=="norm-only": f = L.type in (NM,SM)
        elif mode=="norm-softmax-reduce": f = L.type in (NM,SM,RD)
        elif mode=="attn-core": f = L.type in (SM,NM) or (L.type==MM and "/attn/MatMul" in name and "/qkv/" not in name and "/proj/" not in name)
        elif mode=="attention": f = L.type in (SM,NM) or (L.type==MM and not ("fc1" in name or "fc2" in name or "mlp" in name))
        elif mode=="all": f = L.type in (SM,NM,MM)
        if f:
            L.precision=trt.float32
            for j in range(L.num_outputs): L.set_output_type(j, trt.float32)
            n32+=1
    print(f"DART mixed precision ({mode}): {n32} layers pinned FP32"); a.obey=True
if a.fp32_layers:
    import re as _re; rx=_re.compile(a.fp32_layers); n_pin=0; kinds=collections.Counter()
    for i in range(net.num_layers):
        L=net.get_layer(i)
        if rx.search(L.name):
            if L.type in (trt.LayerType.SHAPE, trt.LayerType.CONSTANT, trt.LayerType.GATHER, trt.LayerType.CONCATENATION) and L.get_output(0).dtype in (trt.int32, trt.int64, trt.bool): continue
            try:
                L.precision=trt.float32
                for j in range(L.num_outputs):
                    if L.get_output(j).dtype in (trt.float32, trt.float16): L.set_output_type(j, trt.float32)
                n_pin+=1; kinds[str(L.type)]+=1
            except Exception as e: pass
    print("pinned %d layers to FP32:"%n_pin, dict(kinds)); a.obey=True
cfg=b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb*(1<<30)))
if not a.no_fp16: cfg.set_flag(trt.BuilderFlag.FP16)
if not a.no_int8: cfg.set_flag(trt.BuilderFlag.INT8)
if a.obey: cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
if a.sparse: cfg.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
if a.timing_cache:
    try: data=open(a.timing_cache,"rb").read()
    except FileNotFoundError: data=b""
    tc=cfg.create_timing_cache(data); cfg.set_timing_cache(tc, False)
t=time.time(); plan=b.build_serialized_network(net,cfg)
assert plan is not None, "build failed"
open(a.plan,"wb").write(memoryview(plan)); print("built in %.0fs: %d MB"%(time.time()-t, plan.nbytes>>20))
if a.timing_cache: open(a.timing_cache,"wb").write(cfg.get_timing_cache().serialize())
try:
    rt=trt.Runtime(logger); eng=rt.deserialize_cuda_engine(plan); insp=eng.create_engine_inspector()
    tally=collections.Counter(); n_int8=0
    for i in range(eng.num_layers):
        raw=insp.get_layer_information(i, trt.LayerInformationFormat.JSON)
        try: info=json.loads(raw)
        except Exception: info=raw
        txt=json.dumps(info) if isinstance(info,dict) else str(info)
        kind="int8" if "Int8" in txt else "fp16" if "Half" in txt else "fp32" if "Float" in txt else "?"
        tally[kind]+=1; n_int8+=(kind=="int8")
    print("layer precision tally:", dict(tally), "| layers touching Int8:", n_int8)
except Exception as e:
    print("inspector skipped:", str(e)[:120])
