"""Energy per frame: run an engine continuously for N seconds while sampling tegrastats; report mean VDD_GPU_SOC / VDD_CPU_CV / VIN_SYS_5V0 power and J/frame."""
import subprocess, time, re, sys, os, numpy as np, json
from trt_util import load_engine, Runner
from preprocess import load_image_tensor
H=os.path.expanduser("~/sam3_orin_agx64_jp72_handoff"); plan=sys.argv[1]; secs=float(sys.argv[2]) if len(sys.argv)>2 else 20
r=Runner(load_engine(plan))
if "images" in r.inputs or "pixel_values" in r.inputs:
    x,_=load_image_tensor(f"{H}/data/images/000000473003.jpg"); feeds={r.inputs[0]:x}
else:   # grounding engines: slice the K=16 golden fixtures to the engine's K
    K=r.bufs[r.inputs[0]][1][0]; gi=lambda n,dt,sh: np.fromfile(f"{H}/goldens/inputs/ground_c16__{n}.bin",dtype=dt).reshape(sh)
    feeds={"img_feat":gi("img_feat",np.float16,(16,256,72,72))[:K],"img_pos":gi("img_pos",np.float16,(16,256,72,72))[:K],"text_feats":np.ascontiguousarray(gi("text_feats",np.float16,(32,16,256))[:,:K,:]),"text_mask":gi("text_mask",np.float32,(16,32))[:K]}
for n in r.inputs:
    buf,shape,dt=r.bufs[n]; buf.upload(np.ascontiguousarray(feeds[n],dtype=dt))
ptrs=[r.bufs[n][0].ptr for n in r.names]
for _ in range(10): r.ctx.execute_v2(ptrs)
from cuda_alloc import DevBuf; DevBuf.sync()
p=subprocess.Popen(["tegrastats","--interval","200"],stdout=subprocess.PIPE,text=True); time.sleep(1.0)
t0=time.perf_counter(); n=0
while time.perf_counter()-t0<secs: r.ctx.execute_v2(ptrs); n+=1
DevBuf.sync(); t1=time.perf_counter(); p.terminate(); out=p.stdout.read()
def mean(key):
    v=[float(m.group(1)) for m in re.finditer(key+r" (\d+)mW",out)]; return np.mean(v)/1000 if v else float("nan")
gpu,cpu,sys5=mean("VDD_GPU_SOC"),mean("VDD_CPU_CV"),mean("VIN_SYS_5V0"); fps=n/(t1-t0)
res={"plan":os.path.basename(os.path.dirname(plan))+"/"+os.path.basename(plan),"frames":n,"ms_per_frame":1000/fps,"W_gpu_soc":gpu,"W_cpu_cv":cpu,"W_sys5v":sys5,"J_per_frame_gpu_soc":gpu/fps,"J_per_frame_sys":sys5/fps}
print(json.dumps(res)); open(os.path.expanduser("~/w8a8/energy.jsonl"),"a").write(json.dumps(res)+"\n")
