"""Assemble reviewed play with 1s context, exact frame mapping and lossless 8-bit YUV."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import av
import numpy as np
import pandas as pd

from shuttlecoach.media import probe,sha256,write_json,read_frame

parser=argparse.ArgumentParser();parser.add_argument('review');parser.add_argument('--out',required=True)
args=parser.parse_args();review=json.loads(Path(args.review).read_text());meta=probe(review['source_video'])
if sha256(meta['path'])!=review['source_sha256']:raise ValueError('Reviewed original source changed.')
intervals=review['reviewed_rallies'];times=np.asarray(meta['times']);rate=Fraction(meta['rate']);tb=Fraction(1,90000)
out=Path(args.out).resolve();out.mkdir(exist_ok=True);video=out/'five-minutes-play-with-context.mp4'
frames=[];rallies=[];previous=-1
for index,rally in enumerate(intervals):
    a,b=rally['start_s'],rally['end_s']
    if not previous<=a<b<=meta['duration_s']:raise ValueError('Review ordered, non-overlapping original intervals.')
    before=max(0,a-1.,(previous+a)/2 if previous>=0 else 0.)
    after=min(meta['duration_s'],b+1.,(b+intervals[index+1]['start_s'])/2 if index+1<len(intervals) else meta['duration_s'])
    first,last=np.searchsorted(times,[before,after])
    live_first,live_last=np.searchsorted(times,[a,b]);base=len(frames)
    if frames and first<=frames[-1]:raise ValueError('Context windows overlap; reduce padding or merge intervals.')
    frames.extend(range(first,last))
    rallies.append(rally|{'start_s':(base+live_first-first)/float(rate),'end_s':(base+live_last-first)/float(rate),
                         'original_start_s':float(times[live_first]),'original_end_s':float(times[live_last])})
    previous=b
with av.open(meta['path']) as source,av.open(str(video),'w') as encoded:
    stream=encoded.add_stream('libx264',rate=rate);stream.width=meta['width'];stream.height=meta['height'];stream.pix_fmt='yuv420p'
    stream.time_base=stream.codec_context.time_base=tb;stream.options={'crf':'0','preset':'veryfast','bf':'0'}
    for field in ['colorspace','color_range','color_primaries','color_trc']:
        setattr(stream.codec_context,field,getattr(source.streams.video[0].codec_context,field))
    for i,f in enumerate(frames):
        # Seek once per segment; H.264 preserves the source's 8-bit YUV conversion.
        if i==0 or f!=frames[i-1]+1:
            source.seek(max(0,int((times[f]+meta['pts_origin_s'])*av.time_base)),backward=True)
            decoder=source.decode(video=0)
        for frame in decoder:
            stamp=float(frame.pts*frame.time_base)-meta['pts_origin_s']
            if stamp<times[f]-.0001:continue
            if abs(stamp-times[f])>.0001:raise ValueError('Assembled frame disagrees with original timestamps.')
            frame.pts=round(Fraction(i,1)/rate/tb);frame.time_base=tb
            for packet in stream.encode(frame):encoded.mux(packet)
            break
        else:raise ValueError('Source ended during assembly.')
        if (i+1)%1800==0:print(f'Assembled {i+1}/{len(frames)} frames',flush=True)
    for packet in stream.encode():encoded.mux(packet)
mapping=out/'source_frame_map.csv'
pd.DataFrame({'clip_frame':range(len(frames)),'original_source_frame':frames,'original_source_time_s':times[frames]}).to_csv(mapping,index=False)
assembled=probe(video)
if assembled['frame_count']!=len(frames):raise ValueError('Assembled video lost source frames.')
checks=[]
def planes(path,index,timeline,origin):
    with av.open(str(path)) as source:
        target=timeline[index]+origin;source.seek(max(0,int(target*av.time_base)),backward=True)
        for frame in source.decode(video=0):
            if abs(float(frame.pts*frame.time_base)-target)<.0001:return frame.to_ndarray(format='yuv420p')
    raise ValueError('Could not verify assembled source pixels.')
for i in np.linspace(0,len(frames)-1,12).round().astype(int):
    original=read_frame(meta['path'],frames[i],meta['times'],meta['pts_origin_s'])
    actual=read_frame(video,int(i),assembled['times'],assembled['pts_origin_s'])
    error=float(np.abs(original.astype(float)-actual.astype(float)).mean());checks.append(error)
    before=planes(meta['path'],frames[i],meta['times'],meta['pts_origin_s'])
    after=planes(video,int(i),assembled['times'],assembled['pts_origin_s'])
    if not np.array_equal(before,after):raise ValueError('Assembly changed the source 8-bit YUV pixels.')
config=json.loads(Path(review['base_config']).read_text())
config.update(video=str(video),source_frame_map={'path':str(mapping),'sha256':sha256(mapping)},
              original_source={k:meta[k] for k in ['path','sha256','duration_s','width','height']},
              rallies=rallies,includes_between_point_pauses=False,focus_player='Gary')
config['calibration']['frame']=int(round(rallies[0]['start_s']*float(rate)))
config.pop('reviewed_rallies',None);config.pop('rally_review',None)
write_json(out/'config.json',config)
cuts=[0,*(np.flatnonzero(np.diff(frames)>1)+1).tolist(),len(frames)]
flight=[[max(rallies[0]['start_s'],a/float(rate)),min(rallies[-1]['end_s'],b/float(rate))] for a,b in zip(cuts,cuts[1:])]
activity=review|{'source_sha256':assembled['sha256'],'original_source_sha256':meta['sha256'],
                 'reviewed_rallies':rallies,'target_flight_intervals':[[a,b] for a,b in flight if a<b],
                 'target_flight_note':'Includes the retained one-second source context around each activity interval; beam association resets between source segments.'}
write_json(out/'activity_review.json',activity)
write_json(out/'assembly.json',{'original_sha256':meta['sha256'],'assembled_sha256':assembled['sha256'],
    'frames':len(frames),'reviewed_play_s':sum(r['end_s']-r['start_s'] for r in rallies),
    'duration_s':assembled['duration_s'],'mean_rgb_conversion_errors':checks,'yuv8_exact':True,'context_s':1.})
print(out/'config.json',flush=True)
