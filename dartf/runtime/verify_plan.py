"""Compare a vision plan against the ORT FP32 truth on image 473003 (fpn_2), print timing."""
import numpy as np, os, sys
from trt_util import load_engine, Runner
from preprocess import load_image_tensor
H=os.path.expanduser("~/sam3_orin_agx64_jp72_handoff"); W=os.path.expanduser("~/w8a8")
golden=sys.argv[2] if len(sys.argv)>2 else f"{W}/exp/bisect/ort_ref_fpn2.npy"; size=int(sys.argv[3]) if len(sys.argv)>3 else 1008
x,_=load_image_tensor(f"{H}/data/images/000000473003.jpg", size); ort32=np.load(golden)
r=Runner(load_engine(sys.argv[1])); o=r({"images":x}, want=["fpn_2"]); y=o["fpn_2"].astype(np.float32)
rel=np.linalg.norm(y-ort32)/np.linalg.norm(ort32); cos=(y.ravel().astype(np.float64)@ort32.ravel().astype(np.float64))/(np.linalg.norm(y)*np.linalg.norm(ort32))
print(f"VERIFY {os.path.basename(sys.argv[1])}: fpn_2 vs ORT-FP32 rel={rel:.4f} cos={cos:.5f} | {r.time({'images':x},10,40):.1f} ms")
