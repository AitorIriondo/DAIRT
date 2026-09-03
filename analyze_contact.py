#!/usr/bin/env python3
"""Hand-object contact analysis over a video.

Runs DART with segmentation masks, then for every frame measures how much each
requested object class overlaps the operator's hand(s). Writes the raw
per-frame overlap to JSON and builds an HTML report with a synced timeline.

The expensive part (inference) and the cheap part (thresholding/smoothing) are
deliberately separate: filter parameters live in the HTML as sliders and are
applied client-side, so retuning never requires re-running the model.

Example:
    PYTHONIOENCODING=utf-8 python analyze_contact.py \\
        --video person_tool_video.mp4 \\
        --classes person hand "power tool" "air hose" \\
        --checkpoint sam3.pt \\
        --trt hf_backbone_fp16.engine \\
        -o contact_report.html
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast


def mask_contact(hand_mask, obj_mask, dilate_px):
    """Overlap between a hand mask and an object mask, normalised by hand area.

    Masks are dilated before intersecting because SAM assigns each contact
    pixel to one object or the other, so a firm grip often produces masks that
    are adjacent rather than overlapping. Without dilation the signal reads
    zero at exactly the moment of contact.

    Returns a float in [0, 1]: the fraction of the hand that is touching.
    """
    if not hand_mask.any():
        return 0.0

    if dilate_px > 0:
        k = np.ones((dilate_px, dilate_px), np.uint8)
        hand_d = cv2.dilate(hand_mask.astype(np.uint8), k, iterations=1)
        obj_d = cv2.dilate(obj_mask.astype(np.uint8), k, iterations=1)
    else:
        hand_d = hand_mask.astype(np.uint8)
        obj_d = obj_mask.astype(np.uint8)

    # Normalise by the *dilated* hand area, not the raw one: dilation grows a
    # small hand mask a lot, so dividing by the raw area lets the ratio exceed 1
    # and inflates exactly the small/distant hands that are least reliable.
    hand_area = int(hand_d.sum())
    if hand_area == 0:
        return 0.0

    inter = int(np.logical_and(hand_d, obj_d).sum())
    return float(inter) / float(hand_area)


def centroid(mask, w, h):
    """Mask centre of mass, normalised to [0, 1] so it is resolution-agnostic."""
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return None
    return [round(m["m10"] / m["m00"] / w, 5), round(m["m01"] / m["m00"] / h, 5)]


def build_html(data, video_path, out_path, embed=False):
    """Write the HTML report. Filtering happens in the browser, not here.

    With embed=True the video is inlined as a base64 data URI, so the file
    opens by double-click with no web server. Browsers treat every file:// page
    as its own origin, so a page loading a sibling .mp4 is blocked as
    cross-origin; a data: URI is part of the document and sidesteps that.
    Base64 costs about 33% size, so compress the clip first.
    """
    payload = json.dumps(data, separators=(",", ":"))

    if embed:
        import base64
        with open(video_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        src = "data:video/mp4;base64," + b64
    else:
        src = os.path.basename(video_path)

    html = HTML_TEMPLATE.replace("__PAYLOAD__", payload)
    html = html.replace("__VIDEO__", src)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


HTML_TEMPLATE = r"""<title>Hand-Tool Contact</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --ink: #16181d; --muted: #6b7280;
    --line: #e3e6ea; --accent: #2f6f4f; --off: #d6d9de;
  }
  :root:not([data-theme="light"]) {
    @media (prefers-color-scheme: dark) {
      --bg: #14161a; --panel: #1c1f25; --ink: #e8eaed; --muted: #9aa1ab;
      --line: #2b2f37; --accent: #4ea87a; --off: #333842;
    }
  }
  :root[data-theme="dark"] {
    --bg: #14161a; --panel: #1c1f25; --ink: #e8eaed; --muted: #9aa1ab;
    --line: #2b2f37; --accent: #4ea87a; --off: #333842;
  }
  body { background: var(--bg); color: var(--ink); margin: 0;
         font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 14px; margin-bottom: 22px; }
  .card { background: var(--panel); border: 1px solid var(--line);
          border-radius: 10px; padding: 16px; margin-bottom: 18px; }
  video { width: 100%; border-radius: 6px; display: block; background: #000; }
  .lane { margin: 14px 0 0; }
  .lane-head { display: flex; justify-content: space-between;
               font-size: 13px; margin-bottom: 5px; }
  .lane-name { font-weight: 600; }
  .lane-stat { color: var(--muted); font-variant-numeric: tabular-nums; }
  .track { position: relative; height: 26px; background: var(--off);
           border-radius: 4px; overflow: hidden; cursor: pointer; }
  .seg { position: absolute; top: 0; bottom: 0; background: var(--accent); }
  .play { position: absolute; top: 0; bottom: 0; width: 2px;
          background: var(--ink); pointer-events: none; }
  .ctrls { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
           gap: 14px; }
  .ctrl label { display: block; font-size: 13px; margin-bottom: 4px; }
  .ctrl span { color: var(--muted); font-variant-numeric: tabular-nums; }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 500; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .scroll { overflow-x: auto; }
  .log-head { display: flex; justify-content: space-between; align-items: center;
              margin-bottom: 10px; }
  .log-head strong { font-size: 15px; }
  button { font: inherit; font-size: 13px; padding: 5px 12px; cursor: pointer;
           color: var(--ink); background: var(--panel);
           border: 1px solid var(--line); border-radius: 6px; }
  button:hover { border-color: var(--accent); }
  ol.log { list-style: none; margin: 0; padding: 0;
           max-height: 340px; overflow-y: auto; }
  ol.log li { display: flex; gap: 12px; padding: 5px 8px; border-radius: 5px;
              cursor: pointer; font-size: 14px; }
  ol.log li:hover { background: var(--off); }
  ol.log .t { color: var(--muted); font-variant-numeric: tabular-nums;
              min-width: 52px; }
  ol.log .dot { width: 9px; height: 9px; border-radius: 50%;
                align-self: center; flex: none; }
  ol.log .held { color: var(--muted); margin-left: auto; padding-left: 12px;
                 font-variant-numeric: tabular-nums; }
  .empty { color: var(--muted); font-size: 14px; padding: 6px 8px; }
  .motion { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--line); }
  .motion-head { font-size: 13px; font-weight: 600; margin-bottom: 9px; }
  .motion-boxes { display: flex; flex-wrap: wrap; gap: 8px; }
  .motion-boxes label { display: inline-flex; align-items: center; gap: 7px;
      font-size: 13px; padding: 5px 11px; border: 1px solid var(--line);
      border-radius: 999px; cursor: pointer; user-select: none; }
  .motion-boxes label:hover { border-color: var(--accent); }
  .motion-boxes input { accent-color: var(--accent); margin: 0; }
  .motion-boxes .swatch { width: 9px; height: 9px; border-radius: 50%; }
  .hint { color: var(--muted); font-size: 12.5px; margin-top: 10px;
          max-width: 62ch; line-height: 1.45; }
</style>

<div class="wrap">
  <h1>Hand-tool contact</h1>
  <div class="sub" id="meta"></div>

  <div class="card">
    <video id="vid" src="__VIDEO__" controls></video>
    <div id="lanes"></div>
  </div>

  <div class="card">
    <div class="ctrls">
      <div class="ctrl">
        <label>Enter contact <span id="lHi"></span></label>
        <input type="range" id="hi" min="0.005" max="0.5" step="0.005" value="0.165">
      </div>
      <div class="ctrl">
        <label>Leave contact <span id="lLo"></span></label>
        <input type="range" id="lo" min="0.000" max="0.5" step="0.005" value="0.165">
      </div>
      <div class="ctrl">
        <label>Bridge gaps <span id="lClose"></span></label>
        <input type="range" id="close" min="0" max="60" step="1" value="15">
      </div>
      <div class="ctrl">
        <label>Drop blips <span id="lOpen"></span></label>
        <input type="range" id="open" min="0" max="60" step="1" value="0">
      </div>
      <div class="ctrl">
        <label>Smoothing <span id="lEma"></span></label>
        <input type="range" id="ema" min="0" max="0.9" step="0.05" value="0.3">
      </div>
      <div class="ctrl">
        <label>Motion window <span id="lMw"></span></label>
        <input type="range" id="mw" min="4" max="60" step="2" value="20">
      </div>
      <div class="ctrl">
        <label>Motion strictness <span id="lMt"></span></label>
        <input type="range" id="mt" min="0" max="0.9" step="0.05" value="0.3">
      </div>
    </div>

    <div class="motion">
      <div class="motion-head">Must move with the hand</div>
      <div id="motionBoxes" class="motion-boxes"></div>
      <div class="hint">Tick an object that should only count as held when it
        travels with the hand. Untick things that stay put when touched, such as
        a table or a fixed machine. Frames where the hand is still are never
        vetoed, since motion cannot tell either way there.</div>
    </div>
  </div>

  <div class="card scroll">
    <table>
      <thead><tr>
        <th>Object</th><th class="num">Contact time</th><th class="num">% of clip</th>
        <th class="num">Grasps</th><th class="num">Mean grasp</th><th class="num">Longest</th>
      </tr></thead>
      <tbody id="stats"></tbody>
    </table>
  </div>

  <div class="card">
    <div class="log-head">
      <strong>Event log</strong>
      <button id="copy">Copy as text</button>
    </div>
    <ol class="log" id="log"></ol>
  </div>
</div>

<script>
const D = __PAYLOAD__;
const fps = D.fps, N = D.n_frames;
// One hue per object class, shared by the timeline lanes and the log dots.
const COLOURS = ["#2f8f5b", "#2b6cb0", "#b7791f", "#8b5cf6", "#c0504d", "#0f9b9b"];
// Motion gating is on by default: most objects a person grasps do move with them.
const MOTION = {};
D.objects.forEach(n => { MOTION[n] = true; });
const HAS_TRACKS = !!(D.obj_xy && D.hand_xy);

document.getElementById("meta").textContent =
  `${D.video} · ${N} frames · ${fps.toFixed(1)} fps · ${(N/fps).toFixed(1)} s · hand class: "${D.hand_class}"`;

// --- motion gating -------------------------------------------------------
// A held object keeps a roughly constant offset from the hand: when the hand
// moves, the object moves with it. A static object (a can on a table that the
// hand merely passes in front of) does not. Over a window we project the
// object's displacement onto the hand's:
//     agreement = (dObj . dHand) / |dHand|^2
// held -> ~1, static -> ~0. When the hand barely moves the test cannot tell,
// so it returns inconclusive and never vetoes on its own.
function motionAgreement(name, win) {
  const O = D.obj_xy && D.obj_xy[name], H = D.hand_xy && D.hand_xy[name];
  const out = new Array(N).fill(null);
  if (!O || !H) return out;
  const half = Math.max(1, Math.round(win / 2));
  for (let i = 0; i < N; i++) {
    const a = Math.max(0, i - half), b = Math.min(N - 1, i + half);
    if (!O[a] || !O[b] || !H[a] || !H[b]) continue;
    const dox = O[b][0] - O[a][0], doy = O[b][1] - O[a][1];
    const dhx = H[b][0] - H[a][0], dhy = H[b][1] - H[a][1];
    const hMag2 = dhx * dhx + dhy * dhy;
    // Hand displacement below ~1% of frame width tells us nothing.
    if (Math.sqrt(hMag2) < 0.01) { out[i] = null; continue; }
    out[i] = (dox * dhx + doy * dhy) / hMag2;
  }
  return out;
}

// --- signal processing: EMA -> hysteresis -> morphological close/open ---
function ema(x, a) {
  if (a <= 0) return x.slice();
  const y = new Array(x.length); let p = x[0] || 0;
  for (let i = 0; i < x.length; i++) { p = a * p + (1 - a) * x[i]; y[i] = p; }
  return y;
}
function hysteresis(x, hi, lo) {
  const b = new Array(x.length); let on = false;
  for (let i = 0; i < x.length; i++) {
    if (!on && x[i] >= hi) on = true;
    else if (on && x[i] < lo) on = false;
    b[i] = on;
  }
  return b;
}
// close = fill false-gaps shorter than w; open = delete true-runs shorter than w
function morph(b, w, fill) {
  if (w <= 0) return b.slice();
  const y = b.slice();
  let i = 0;
  while (i < y.length) {
    if (y[i] === !fill) {
      let j = i; while (j < y.length && y[j] === !fill) j++;
      if (j - i < w && (i > 0 || !fill === false) ) {
        // only bridge/remove interior runs, not the leading/trailing edge
        if (i > 0 && j < y.length) for (let k = i; k < j; k++) y[k] = fill;
      }
      i = j;
    } else i++;
  }
  return y;
}
function segments(b) {
  const out = []; let s = -1;
  for (let i = 0; i < b.length; i++) {
    if (b[i] && s < 0) s = i;
    else if (!b[i] && s >= 0) { out.push([s, i]); s = -1; }
  }
  if (s >= 0) out.push([s, b.length]);
  return out;
}

function compute() {
  const hi = +hiEl.value, lo = Math.min(+loEl.value, +hiEl.value),
        cw = +closeEl.value, ow = +openEl.value, a = +emaEl.value,
        mw = +mwEl.value, mt = +mtEl.value;
  lMw.textContent = mw + " fr (" + (mw / fps).toFixed(2) + " s)";
  lMt.textContent = mt.toFixed(2);
  lHi.textContent = hi.toFixed(3); lLo.textContent = lo.toFixed(3);
  lClose.textContent = cw + " fr (" + (cw/fps).toFixed(2) + " s)";
  lOpen.textContent = ow + " fr (" + (ow/fps).toFixed(2) + " s)";
  lEma.textContent = a.toFixed(2);

  const lanes = document.getElementById("lanes");
  const stats = document.getElementById("stats");
  lanes.innerHTML = ""; stats.innerHTML = "";
  const events = [];

  for (const name of D.objects) {
    let sig = ema(D.overlap[name], a);
    let b = hysteresis(sig, hi, lo);
    if (MOTION[name] && HAS_TRACKS) {
      // Veto contact only where the hand moved and the object did not follow.
      const agr = motionAgreement(name, mw);
      for (let i = 0; i < b.length; i++) {
        if (b[i] && agr[i] !== null && agr[i] < mt) b[i] = false;
      }
    }
    b = morph(b, cw, true);    // close: bridge dropouts inside a grasp
    b = morph(b, ow, false);   // open: delete short phantom grasps
    const segs = segments(b);

    const lane = document.createElement("div");
    lane.className = "lane";
    const frames = b.reduce((s, v) => s + (v ? 1 : 0), 0);
    lane.innerHTML =
      `<div class="lane-head"><span class="lane-name">${name}</span>` +
      `<span class="lane-stat">${(frames/fps).toFixed(1)} s · ${segs.length} grasps</span></div>`;
    const tr = document.createElement("div");
    tr.className = "track";
    for (const [s, e] of segs) {
      const d = document.createElement("div");
      d.className = "seg";
      d.style.left = (100 * s / N) + "%";
      d.style.width = (100 * (e - s) / N) + "%";
      d.style.background = COLOURS[D.objects.indexOf(name) % COLOURS.length];
      tr.appendChild(d);
    }
    const ph = document.createElement("div");
    ph.className = "play"; ph.style.left = "0%";
    tr.appendChild(ph);
    tr.addEventListener("click", ev => {
      const r = tr.getBoundingClientRect();
      vid.currentTime = ((ev.clientX - r.left) / r.width) * (N / fps);
    });
    lane.appendChild(tr);
    lanes.appendChild(lane);

    for (const [s, e] of segs) {
      const colour = COLOURS[D.objects.indexOf(name) % COLOURS.length];
      events.push({f: s, kind: "grabs", name, colour, held: null});
      events.push({f: e, kind: "releases", name, colour, held: (e - s) / fps});
    }

    const durs = segs.map(([s, e]) => (e - s) / fps);
    const row = document.createElement("tr");
    row.innerHTML =
      `<td>${name}</td>` +
      `<td class="num">${(frames/fps).toFixed(1)} s</td>` +
      `<td class="num">${(100*frames/N).toFixed(1)}%</td>` +
      `<td class="num">${segs.length}</td>` +
      `<td class="num">${durs.length ? (durs.reduce((x,y)=>x+y,0)/durs.length).toFixed(2) : "0.00"} s</td>` +
      `<td class="num">${durs.length ? Math.max(...durs).toFixed(2) : "0.00"} s</td>`;
    stats.appendChild(row);
  }

  renderLog(events);
}

function tc(frame) {
  const t = frame / fps;
  const m = Math.floor(t / 60), sec = t - m * 60;
  return String(m).padStart(2, "0") + ":" + sec.toFixed(1).padStart(4, "0");
}

function logLines(events) {
  return events.map(e => {
    const held = e.held !== null ? `  (held ${e.held.toFixed(1)}s)` : "";
    return `${tc(e.f)}  Person ${e.kind} the ${e.name}${held}`;
  });
}

function renderLog(events) {
  events.sort((a, b) => a.f - b.f || (a.kind === "releases" ? -1 : 1));
  const log = document.getElementById("log");
  log.innerHTML = "";

  if (!events.length) {
    log.innerHTML = '<div class="empty">No contact events at these settings.</div>';
    LAST_LINES = [];
    return;
  }

  const lines = logLines(events);
  events.forEach((e, i) => {
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="t">${tc(e.f)}</span>` +
      `<span class="dot" style="background:${e.colour}"></span>` +
      `<span>Person <strong>${e.kind}</strong> the ${e.name}</span>` +
      (e.held !== null ? `<span class="held">held ${e.held.toFixed(1)}s</span>` : "");
    li.title = "Jump to " + tc(e.f);
    li.addEventListener("click", () => { vid.currentTime = e.f / fps; });
    log.appendChild(li);
  });
  LAST_LINES = lines;
}

let LAST_LINES = [];

const vid = document.getElementById("vid");
const hiEl = document.getElementById("hi"), loEl = document.getElementById("lo"),
      closeEl = document.getElementById("close"), openEl = document.getElementById("open"),
      emaEl = document.getElementById("ema");
const mwEl = document.getElementById("mw"), mtEl = document.getElementById("mt");
const lMw = document.getElementById("lMw"), lMt = document.getElementById("lMt");
const lHi = document.getElementById("lHi"), lLo = document.getElementById("lLo"),
      lClose = document.getElementById("lClose"), lOpen = document.getElementById("lOpen"),
      lEma = document.getElementById("lEma");

document.getElementById("copy").addEventListener("click", async ev => {
  const txt = LAST_LINES.join(String.fromCharCode(10));
  try {
    await navigator.clipboard.writeText(txt);
    ev.target.textContent = "Copied";
  } catch (err) {
    // clipboard API needs a secure context; file:// pages fall back to select-all
    const ta = document.createElement("textarea");
    ta.value = txt; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); ev.target.textContent = "Copied"; }
    catch (e2) { ev.target.textContent = "Press Ctrl+C"; ta.focus(); return; }
    document.body.removeChild(ta);
  }
  setTimeout(() => { ev.target.textContent = "Copy as text"; }, 1500);
});

function buildMotionBoxes() {
  const box = document.getElementById("motionBoxes");
  box.innerHTML = "";
  if (!HAS_TRACKS) {
    box.innerHTML = '<div class="empty">This report predates motion tracking. ' +
                    'Re-run the analysis to enable it.</div>';
    return;
  }
  D.objects.forEach((name, i) => {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = MOTION[name];
    cb.addEventListener("change", () => { MOTION[name] = cb.checked; compute(); });
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = COLOURS[i % COLOURS.length];
    lab.appendChild(cb); lab.appendChild(sw);
    lab.appendChild(document.createTextNode(name));
    box.appendChild(lab);
  });
}
buildMotionBoxes();

for (const el of [hiEl, loEl, closeEl, openEl, emaEl, mwEl, mtEl]) el.addEventListener("input", compute);
vid.addEventListener("timeupdate", () => {
  const pct = 100 * vid.currentTime / (N / fps);
  document.querySelectorAll(".play").forEach(p => p.style.left = pct + "%");
});
compute();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None,
                    help="Input video. Required unless --from-json is given.")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="Classes to detect. Must include the hand class. "
                         "Required unless --from-json is given.")
    ap.add_argument("--hand-class", default="hand",
                    help="Class treated as the hand (default: hand)")
    ap.add_argument("--contact-classes", nargs="+", default=None,
                    help="Classes to test against the hand. Defaults to every "
                         "detected class except the hand class and 'person'.")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--trt", default=None, help="TRT backbone engine")
    ap.add_argument("--imgsz", type=int, default=1008)
    ap.add_argument("--confidence", type=float, default=0.4)
    ap.add_argument("--nms", type=float, default=0.7)
    ap.add_argument("--dilate", type=int, default=13,
                    help="Mask dilation in px before intersecting (default: 25). "
                         "SAM assigns contact pixels exclusively, so a gripping "
                         "hand and tool have ADJACENT masks: raw intersection is "
                         "0 even during a firm grip, so this must be > 0.")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--work-width", type=int, default=640, metavar="PX",
                    help="Width at which contact geometry is computed (default: "
                         "640). Masks are downscaled to this first, which keeps "
                         "memory bounded and makes --dilate independent of the "
                         "source resolution.")
    ap.add_argument("--max-instances", type=int, default=8, metavar="N",
                    help="Most instances per class per frame, highest score "
                         "first (default: 8). Stops a runaway prompt exploding "
                         "the pair loop.")
    ap.add_argument("--max-jump", type=float, default=0.08, metavar="FRAC",
                    help="Largest centroid move, as a fraction of frame size, "
                         "accepted between analysed frames when following one "
                         "object for motion gating (default: 0.08).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Analyse every Nth frame (default: 1)")
    ap.add_argument("-o", "--output", default="contact_report.html")
    ap.add_argument("--json", default=None,
                    help="Where to write raw per-frame data (default: <output>.json)")
    ap.add_argument("--report-video", default=None, metavar="FILE",
                    help="Video the HTML should display. Defaults to --video, "
                         "i.e. the raw input. Point this at the mask-annotated "
                         "render so the report shows what was measured.")
    ap.add_argument("--embed-video", action="store_true",
                    help="Inline the video as base64 so the HTML opens by "
                         "double-click with no web server. Compress the clip "
                         "first: base64 adds ~33%.")
    ap.add_argument("--from-json", default=None, metavar="FILE",
                    help="Rebuild the HTML from an existing JSON dump without "
                         "re-running inference. Nothing else is needed.")
    args = ap.parse_args()

    # Regenerating the report is pure presentation: no model, no GPU, instant.
    if args.from_json:
        with open(args.from_json, encoding="utf-8") as f:
            data = json.load(f)
        # --video may point the report at a different rendering of the same
        # clip (e.g. the mask-annotated output) without redoing the analysis.
        if args.video:
            data["video"] = os.path.basename(args.video)
        vsrc = args.video or data["video"]
        build_html(data, vsrc, args.output, embed=args.embed_video)
        print(f"Report rebuilt from {args.from_json}: {args.output}")
        return

    if not args.video or not args.classes:
        ap.error("--video and --classes are required (unless --from-json)")

    if args.hand_class not in args.classes:
        ap.error(f"--classes must include the hand class {args.hand_class!r}")

    contact_classes = args.contact_classes
    if contact_classes is None:
        contact_classes = [c for c in args.classes
                           if c != args.hand_class and c != "person"]
    if not contact_classes:
        ap.error("no contact classes: add object classes beyond the hand class")

    unknown = [c for c in contact_classes if c not in args.classes]
    if unknown:
        ap.error(f"--contact-classes not in --classes: {unknown}")

    if args.imgsz % 14 != 0:
        ap.error(f"--imgsz must be divisible by 14, got {args.imgsz}")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Detecting: {args.classes}")
    print(f"Contact vs {args.hand_class!r}: {contact_classes}")

    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        device="cuda",
        eval_mode=True,
        resolution=args.imgsz,
    )
    predictor = Sam3MultiClassPredictorFast(
        model,
        device="cuda",
        resolution=args.imgsz,
        use_fp16=True,
        detection_only=False,          # masks are the whole point here
        trt_engine_path=args.trt,
    )
    predictor.set_classes(args.classes)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    work_w = min(args.work_width, src_w)
    work_h = max(1, round(src_h * work_w / src_w))
    print(f"Contact geometry at {work_w}x{work_h} (source {src_w}x{src_h}), "
          f"dilate {args.dilate}px, max {args.max_instances} instances/class")

    overlap = {c: [] for c in contact_classes}
    # Centroid tracks for motion gating: a genuinely held object keeps a roughly
    # constant offset from the hand, while a static one does not.
    obj_xy = {c: [] for c in contact_classes}
    hand_xy = {c: [] for c in contact_classes}
    obj_prev = {c: None for c in contact_classes}   # last associated centroid
    n_done = 0
    t0 = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok or (args.max_frames and n_done >= args.max_frames):
            break
        if args.stride > 1 and n_done % args.stride != 0:
            n_done += 1
            for c in contact_classes:
                overlap[c].append(overlap[c][-1] if overlap[c] else 0.0)
                obj_xy[c].append(obj_xy[c][-1] if obj_xy[c] else None)
                hand_xy[c].append(hand_xy[c][-1] if hand_xy[c] else None)
            continue

        # Must be PIL: set_image() reads numpy sizes as image.shape[-2:], i.e.
        # CHW. An OpenCV HWC frame would be read as (width, channels) and the
        # masks would come back scaled to a 3-pixel-wide image.
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = predictor.predict_image(
            pil,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms,
        )

        names = res["class_names"]
        masks = res["masks"]
        hands, objs = [], {c: [] for c in contact_classes}
        if masks is not None:
            # Downscale every mask to a fixed working width before any contact
            # maths. Contact is a coarse geometric signal, so full resolution
            # buys nothing and costs a great deal: at 1920x1080 with ~10
            # detections per frame this loop churned ~100 MB per frame through
            # cv2.dilate and could exhaust system RAM on a long clip.
            # Working at a fixed width also makes --dilate resolution-independent.
            keep = {}
            for i, nm in enumerate(names):
                if nm != args.hand_class and nm not in objs:
                    continue
                keep.setdefault(nm, []).append((float(res["scores"][i]), i))

            for nm, items in keep.items():
                # Cap instances per class: a runaway prompt (e.g. "metal bar" at
                # 120 dets/frame) must not be able to blow up the pair loop.
                items.sort(reverse=True)
                for _, i in items[:args.max_instances]:
                    m8 = masks[i].cpu().numpy().astype(np.uint8)
                    if m8.shape[1] != work_w:
                        m8 = cv2.resize(m8, (work_w, work_h),
                                        interpolation=cv2.INTER_NEAREST)
                    m = m8.astype(bool)
                    if nm == args.hand_class:
                        hands.append(m)
                    else:
                        objs[nm].append(m)

        # Per class, keep the strongest hand-object overlap in this frame.
        # Any hand touching any instance counts, so track identity is not needed.
        h_frame, w_frame = work_h, work_w
        for c in contact_classes:
            best = 0.0
            best_pair = None
            for h in hands:
                for o in objs[c]:
                    v = mask_contact(h, o, args.dilate)
                    if v > best:
                        best, best_pair = v, (h, o)
            overlap[c].append(best)

            # Record a centroid pair for motion gating.
            #
            # The object centroid must follow ONE physical instance over time.
            # Picking the best-overlapping instance each frame would make the
            # track follow the hand by construction, so a static object would
            # look like it moves with the hand and the motion test would be
            # circular. Instead associate to the previous frame's centroid by
            # nearest neighbour, seeding with the largest instance.
            o = None
            if objs[c]:
                cands = [(centroid(m, w_frame, h_frame), m) for m in objs[c]]
                cands = [(ct, m) for ct, m in cands if ct is not None]
                if cands:
                    prev = obj_prev[c]
                    if prev is None:
                        o_ct, o = max(cands, key=lambda t: t[1].sum())
                    else:
                        o_ct, o = min(
                            cands,
                            key=lambda t: (t[0][0] - prev[0]) ** 2
                                          + (t[0][1] - prev[1]) ** 2,
                        )
                        # Reject teleports. An open-vocabulary prompt fires on
                        # lookalikes elsewhere in the scene ("soda can" also
                        # matches bottles on a counter), and without this the
                        # track hops onto one near the hand and never returns.
                        d2 = ((o_ct[0] - prev[0]) ** 2 + (o_ct[1] - prev[1]) ** 2)
                        if d2 > args.max_jump ** 2:
                            o_ct, o = prev, None
                    obj_prev[c] = o_ct

            # The hand is whichever is closest to that object, so the pair is
            # the one whose relative motion actually matters.
            h = None
            if hands and obj_prev[c] is not None:
                hc = [(centroid(m, w_frame, h_frame), m) for m in hands]
                hc = [(ct, m) for ct, m in hc if ct is not None]
                if hc:
                    ref = obj_prev[c]
                    _, h = min(
                        hc,
                        key=lambda t: (t[0][0] - ref[0]) ** 2
                                      + (t[0][1] - ref[1]) ** 2,
                    )

            obj_xy[c].append(obj_prev[c] if o is not None else None)
            hand_xy[c].append(centroid(h, w_frame, h_frame) if h is not None else None)

        n_done += 1
        if n_done % 60 == 0:
            el = time.time() - t0
            print(f"  {n_done}/{total} frames  ({n_done/el:.1f} fps)")

    cap.release()
    elapsed = time.time() - t0
    print(f"Done: {n_done} frames in {elapsed:.1f}s ({n_done/elapsed:.1f} fps)")

    data = {
        "video": os.path.basename(args.video),
        "fps": fps,
        "n_frames": n_done,
        "hand_class": args.hand_class,
        "objects": contact_classes,
        "dilate_px": args.dilate,
        "work_size": [work_w, work_h],
        "confidence": args.confidence,
        "overlap": {c: [round(v, 4) for v in overlap[c]] for c in contact_classes},
        "obj_xy": {c: obj_xy[c] for c in contact_classes},
        "hand_xy": {c: hand_xy[c] for c in contact_classes},
    }

    # A class that overlaps the hand in almost every frame at near-full value is
    # not "in contact" with the hand, it IS the hand under another name (glove,
    # mitten, sleeve). Flag it rather than letting it sit in the report as a
    # saturated lane that looks like a perfect result.
    for c in contact_classes:
        v = overlap[c]
        strong = sum(1 for x in v if x >= 0.95)
        if v and strong / len(v) > 0.5:
            print(f"WARNING: {c!r} overlaps {args.hand_class!r} at >=0.95 in "
                  f"{100*strong/len(v):.0f}% of frames. It probably segments the "
                  f"same object as the hand, so this lane measures self-overlap, "
                  f"not contact. Consider dropping it from --contact-classes.")

    json_path = args.json or (os.path.splitext(args.output)[0] + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Raw per-frame overlap: {json_path}")

    build_html(data, args.report_video or args.video, args.output,
               embed=args.embed_video)
    print(f"Report: {args.output}")
    if not args.embed_video:
        print("  (keep the .mp4 beside the .html, and serve over http:// —"
              " browsers block file:// pages from loading sibling video)")


if __name__ == "__main__":
    main()
