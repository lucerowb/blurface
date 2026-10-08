# Blurface

Blur every face in a video except the people you choose to keep. Processing stays on your computer.

Pick a video, analyze the faces, tick the people who should remain visible, and export. Everyone else is blurred for the whole clip, including short gaps where the detector misses a frame. The window shows the input file size and an estimated output size, and you can compress the export before you save it.

## Privacy

- Video is read and written on this machine. Nothing is uploaded.
- The first run downloads two small OpenCV face models (YuNet and SFace, about 38 MB together) into `~/.cache/blurface`.
- Face detection can miss someone who is tiny, turned away, or heavily blocked. Scrub the exported video before you share it.

## Install

Python 3.10 or newer, plus Tkinter.

```bash
pip install -r requirements.txt
```

On macOS with Homebrew Python, Tkinter is a separate package. Match it to your Python version:

```bash
brew install python-tk@3.12
```

`imageio-ffmpeg` supplies the ffmpeg binary used to compress the video and keep the audio. A system `ffmpeg` on your `PATH` is used if that package is missing.

## Run

```bash
python blur_faces.py
```

Or, after `pip install .`:

```bash
blurface
```

1. Choose a video. The input card shows resolution, duration, and file size.
2. Set compression. **Small** and lower resolutions make a smaller file. **Max** keeps more detail. The output card updates the estimate as you change it.
3. Analyze faces.
4. Tick **Keep visible** for the people who should stay sharp. Unchecked people are blurred.
5. Export. When it finishes, the output card switches from the estimate to the saved file size.

| Compression | What you get |
| --- | --- |
| Small | Smallest file. Fine for sharing when detail is secondary. |
| Compact | Default. Usually much smaller than a camera original. |
| High | Clearer faces that you kept, larger file. |
| Max | Highest quality export. |
| 1080p / 720p / 480p | Shrinks the long edge when the source is larger. |

Blur styles are a strong blur, pixelation, or a black box. Blur shape can be Auto, circle, ellipse, or rectangle. Auto uses a circle when the face is square and an ellipse otherwise, and the edge fades into the picture. A hand, phone, or bag under a real face is left alone. Analyze the video again after updating so the stricter detector is used.

## How it works

1. YuNet finds a face box on every frame. SFace turns a clear detection into a fingerprint.
2. Overlapping boxes are linked into tracks. Tracks with similar fingerprints are grouped into people.
3. On export, blur boxes are filled across short detector gaps and extended slightly before and after each appearance.
4. The blurred frames are encoded with H.264 at the CRF you chose, and the original audio is attached as AAC.

Raise **Same-person strictness** and click **Update groups** if two people were merged, or if one person was split into two groups. Splitting one person into two groups is the safer mistake: you can tick both.

The window scrolls. The export bar stays pinned at the bottom.

## Installers

GitHub Actions builds a Windows exe and a macOS dmg when you push a tag such as `v1.0.0`, or when you run the **Build installers** workflow by hand. Download them from the workflow artifacts, or from the GitHub release created for that tag.

The builds are not notarized. macOS may ask you to open the app from the context menu the first time.

## Development

```bash
python -m unittest discover -s tests -v
```

The app is one file, `blur_faces.py`, so the detection, tracking, and window stay easy to read together.

## License

[MIT](LICENSE)
