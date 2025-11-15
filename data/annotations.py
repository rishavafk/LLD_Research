# annotations.py - helper to map annotation files
import os, glob
from typing import Dict

def build_annotation_mapping(gt_root: str) -> Dict[str,str]:
    ann_map = {}
    # search class folders
    for cls_dir in glob.glob(os.path.join(gt_root, "*")):
        if not os.path.isdir(cls_dir): continue
        for txt in glob.glob(os.path.join(cls_dir, "*.txt")):
            base = os.path.basename(txt)
            key = os.path.splitext(base)[0]   # removes final .txt
            ann_map[key] = txt
    # stray txt directly in gt_root
    for txt in glob.glob(os.path.join(gt_root, "*.txt")):
        base = os.path.basename(txt)
        key = os.path.splitext(base)[0]
        ann_map[key] = txt
    print(f"[ANN MAP] {len(ann_map)} annotations found under {gt_root}")
    return ann_map

def parse_exdark_annotation(txt_path):
    """
    ExDark annotation parsing. Returns list of (cls_str, l, t, w, h)
    where l,t,w,h are floats (normalized or pixels) as present in file.
    """
    try:
        s = open(txt_path, "r", errors="ignore").read().strip()
    except:
        return []
    if len(s) <= 16:
        return []
    s = s[16:].strip()
    parts = s.split()
    out=[]
    i=0
    while i < len(parts):
        cls = parts[i]; i+=1
        if i+3 >= len(parts): break
        l = float(parts[i]); t = float(parts[i+1]); w = float(parts[i+2]); h = float(parts[i+3])
        i+=4
        out.append((cls,l,t,w,h))
        i += 7  # skip extra metadata if present
    return out
