"""Make source-only sheets for manual interval selection and independent foot labels."""
import argparse
from pathlib import Path
import av
from PIL import Image,ImageDraw

parser=argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("--seconds",type=float,nargs="+",required=True)
parser.add_argument("--out",required=True)
parser.add_argument("--width",type=int,default=640)
parser.add_argument("--columns",type=int,default=3)
args=parser.parse_args()
targets=iter(sorted(args.seconds))
target=next(targets)
images=[]
with av.open(args.video) as container:
    origin=None
    for index,frame in enumerate(container.decode(video=0)):
        stamp=float(frame.pts*frame.time_base)
        if origin is None:origin=stamp
        stamp-=origin
        if stamp+0.0001>=target:
            image=frame.to_image()
            image.thumbnail((args.width,round(args.width*9/16)))
            images.append((image,f"frame {index} | source {stamp:.3f}s"))
            target=next(targets,None)
            if target is None:break
height=round(args.width*9/16)+24
sheet=Image.new("RGB",(args.columns*args.width,((len(images)+args.columns-1)//args.columns)*height))
draw=ImageDraw.Draw(sheet)
for index,(image,label) in enumerate(images):
    x,y=(index%args.columns)*args.width,(index//args.columns)*height
    sheet.paste(image,(x,y))
    draw.text((x+8,y+height-20),label,fill="white")
sheet.save(args.out)
print(args.out)
