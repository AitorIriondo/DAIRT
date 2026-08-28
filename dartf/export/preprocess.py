"""Exact port of client/preprocess_image.py (Pillow bilinear 1008x1008, uint8/127.5-1)."""
import numpy as np
from PIL import Image
SIZE=1008
def load_image_tensor(path, size=None):
    SIZE_=size or SIZE
    with Image.open(path) as src:
        im=src.convert("RGB"); orig=im.size
        if im.size!=(SIZE_,SIZE_): im=im.resize((SIZE_,SIZE_), resample=Image.Resampling.BILINEAR)
        px=np.asarray(im,dtype=np.uint8)
    t=np.ascontiguousarray(px.transpose(2,0,1),dtype=np.float32); t*=np.float32(1.0/127.5); t-=np.float32(1.0)
    return t[None], orig  # [1,3,1008,1008], (w,h)
