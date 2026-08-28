# Media

`build_explainer.py` rebuilds the whole explainer video reproducibly: it runs `../demo/run_video.py` on every input video (prompts are the underscore separated words of the file name; each clip is cut at 15 s, opens with a with/without wipe of the overlay and carries the lower third (first clip only) for its first 8 s), renders the slide video with `make_media.py`, concatenates the clips and the slides, and mixes the background music with fades:

```bash
python build_explainer.py --videos videos/ --out out/ --music music.mp3 --runner-args "--assets assets --name 'Your Name' --affiliation 'Your lab'"
```

`make_media.py <out_dir> [--video]` renders the slide set (`dartf_slides.pdf`) and the slide video alone (`--preview 0,3` writes PNG previews of scenes). Text is set in Lato, equations with matplotlib mathtext; blocks are laid out by their rendered height so a scene that would overflow the page is reported at startup. The credits scene is generated from `../docs/references.bib`, which `../docs/check_references.py` fetches and verifies from arXiv.
