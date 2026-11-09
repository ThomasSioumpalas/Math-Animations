"""Contact sheet of a video (needs: uv run --with av --with pillow): python tools/contact_sheet.py <video.mp4> <out.png> <t1,t2,...> (seconds) [cols]"""
import sys

import av
from PIL import Image, ImageDraw

path, out, times = sys.argv[1], sys.argv[2], [float(t) for t in sys.argv[3].split(",")]
cols = int(sys.argv[4]) if len(sys.argv) > 4 else 3
w, h = 480, 270
c = av.open(path)
s = c.streams.video[0]
want, got = sorted(times), []
for fr in c.decode(s):
    while want and fr.time >= want[0]:
        got.append((want.pop(0), fr.to_image().resize((w, h))))
    if not want:
        break
rows = (len(got) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * h))
d = ImageDraw.Draw(sheet)
for k, (t, im) in enumerate(got):
    x, y = (k % cols) * w, (k // cols) * h
    sheet.paste(im, (x, y))
    d.text((x + 4, y + 4), f"{t:.0f}s", fill=(255, 80, 80))
sheet.save(out)
print("saved", out, len(got), "frames")
