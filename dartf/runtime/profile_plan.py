import json, re, collections, sys, subprocess, os
plan=sys.argv[1]; tag=sys.argv[2]
subprocess.run(["/usr/src/tensorrt/bin/trtexec",f"--loadEngine={plan}","--dumpProfile","--separateProfileRun","--iterations=30","--warmUp=1000","--duration=0",f"--exportProfile=/tmp/{tag}_profile.json"],capture_output=True)
rows=[r for r in json.load(open(f"/tmp/{tag}_profile.json")) if "name" in r and "averageMs" in r]; tot=sum(r["averageMs"] for r in rows)
fam=collections.Counter(); cnt=collections.Counter()
for r in rows:
    n=r["name"]; nl=n.lower()
    if "_gemm_mha" in n or "mha" in nl: f="attention(fused MHA)"
    elif "matmul" in nl or "__myl_fc" in nl or "gemm" in nl: f="GEMM"
    elif "conv" in nl: f="conv(neck)"
    elif "pad" in nl: f="pad"
    elif re.search(r"resh|tran|reformat|cast|copy|slic|concat|gath",nl): f="reformat/shuffle"
    elif re.search(r"norm|reducemean|pow|sqrt",nl): f="norm"
    elif re.search(r"gelu|erf|mul|add|sub|div|softmax|quant|neg",nl): f="elementwise/quant"
    else: f="other"
    fam[f]+=r["averageMs"]; cnt[f]+=1
print(f"{tag}: {len(rows)} layers, {tot:.1f} ms")
for f,v in fam.most_common(): print(f"  {f:22s} {v:6.1f} ms {100*v/tot:5.1f}% ({cnt[f]})")
print("  top:", [(round(r['averageMs'],2), r['name'][:70]) for r in sorted(rows,key=lambda r:-r['averageMs'])[:6]])
