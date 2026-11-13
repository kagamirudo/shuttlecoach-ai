"""Create 16 source-only reference frames, independent of detector output."""
import json
from pathlib import Path
import sys
import av
import pandas as pd
from PIL import Image,ImageDraw

run=Path(sys.argv[1])
manifest=json.loads((run/"analysis.json").read_text())
frames=pd.read_csv(run/"reference_template.csv").source_frame.unique().tolist()
indices=[0,7,14,21,28,35,42,49,50,57,64,71,78,85,92,99]
selected=[frames[i] for i in indices if i<len(frames)]
collected={}
with av.open(manifest["video"]["path"]) as container:
    for index,frame in enumerate(container.decode(video=0)):
        if index in selected:collected[index]=frame.to_image().crop((0,200,1280,720)).resize((640,260))
        if index>=max(selected):break
for page in range(2):
    sheet=Image.new("RGB",(1280,1128))
    draw=ImageDraw.Draw(sheet)
    for cell,frame in enumerate(selected[page*8:(page+1)*8]):
        x,y=cell%2*640,cell//2*282
        sheet.paste(collected[frame],(x,y))
        draw.text((x+8,y+264),f"frame {frame} | source {manifest['video']['times'][frame]:.3f}s | crop x0,y200 scale 0.5",fill="white")
    sheet.save(run/f"pilot-source-{page+1}.png")
(run/"pilot_frames.json").write_text(json.dumps(selected))
print(selected)
