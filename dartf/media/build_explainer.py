"""Reproducible build of the DARTF explainer video: demo clips (run_video.py on each input video, prompts from the file name)
-> slide video (make_media.py) -> concatenation -> background music with fades.
usage: python build_explainer.py --videos <dir> --out <dir> --music <mp3> [--runner ../demo/run_video.py] [--runner-args "--assets assets"]
       [--order a,b,c] [--duration 15] [--slides <existing mp4>] [--fps 30] [--music-volume 0.35]
Clips are named after the input files; only the first clip carries the lower third (slid in after 0.5 s, gone by 8 s); every clip is cut at --duration seconds.
The prompts of a clip are the underscore separated words of its file name (vehicle_person.mp4 -> "vehicle,person")."""
import os, sys, json, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--videos", required=True); ap.add_argument("--out", required=True); ap.add_argument("--music", default=None)
ap.add_argument("--runner", default=os.path.join(HERE, "..", "demo", "run_video.py")); ap.add_argument("--runner-args", default=""); ap.add_argument("--python", default=sys.executable)
ap.add_argument("--order", default=None, help="comma list of file stems in play order (default: alphabetical)"); ap.add_argument("--duration", type=float, default=15.0)
ap.add_argument("--slides", default=None, help="existing slide video; rendered with make_media.py when omitted"); ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--music-volume", type=float, default=0.35); ap.add_argument("--skip-demos", action="store_true"); ap.add_argument("--final", default="dartf_explainer.mp4")
a = ap.parse_args(); os.makedirs(os.path.join(a.out, "demos"), exist_ok=True)
def run(cmd, **kw): print("+", " ".join(cmd), flush=True); subprocess.run(cmd, check=True, **kw)
def duration(path): return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip())
files = {os.path.splitext(f)[0]: os.path.join(a.videos, f) for f in os.listdir(a.videos) if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))}
order = a.order.split(",") if a.order else sorted(files); missing = [s for s in order if s not in files]; assert not missing, f"missing videos: {missing}"
clips = []
for i, stem in enumerate(order):
    out = os.path.join(a.out, "demos", f"demo_{stem}.mp4"); prompts = ",".join(w for w in stem.split("_") if w)
    if not a.skip_demos:
        cmd = [a.python, a.runner, files[stem], out, prompts, "--duration", str(a.duration), "--fps-out", str(a.fps)] + (["--lt-slide"] if i == 0 else ["--no-lower-third"]) + a.runner_args.split()
        run(cmd)
    clips.append(out)
slides = a.slides
if slides is None:
    run([a.python, os.path.join(HERE, "make_media.py"), a.out, "--video", "--fps", str(a.fps)]); slides = os.path.join(a.out, "dartf_explainer_slides.mp4")
    os.replace(os.path.join(a.out, "dartf_explainer.mp4"), slides)
lst = os.path.join(a.out, "concat.txt"); open(lst, "w").write("".join(f"file '{os.path.abspath(p)}'\n" for p in clips + [slides]))
silent = os.path.join(a.out, "explainer_silent.mp4")
run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium", "-r", str(a.fps), "-an", silent])
final = os.path.join(a.out, a.final)
if a.music:
    D = duration(silent); fade = 4.0
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-stream_loop", "-1", "-i", a.music, "-filter_complex", f"[1:a]volume={a.music_volume},afade=t=in:st=0:d=2,afade=t=out:st={D - fade:.2f}:d={fade}[a]",
         "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{D:.3f}", final])
    os.remove(silent)
else: os.replace(silent, final)
os.remove(lst); print(json.dumps({"final": final, "seconds": duration(final), "clips": clips, "slides": slides}))
