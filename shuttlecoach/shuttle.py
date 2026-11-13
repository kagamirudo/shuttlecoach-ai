"""TrackNetV3 adapter: source-aligned frames, bounded memory, no invented gaps."""
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

import av
import cv2
import numpy as np
import pandas as pd

from .media import read_frame, sha256, verify_overlay, write_json
from .paths import ROOT, MODELS

TRACKNET = ROOT / "vendor" / "TrackNetV3"


def weighted_heatmaps(model, batches, frame_count, device, weights):
    """Average overlapping upstream predictions; normalize the shorter edge histories."""
    import torch
    total=None;normalizer=torch.zeros(frame_count)
    for indices,x in batches:
        with torch.inference_mode():heat=model(x.float().to(device)).cpu()
        if total is None:total=torch.zeros((frame_count,*heat.shape[-2:]))
        for ids,prediction in zip(indices,heat):
            frames=ids[:,1].long()
            total.index_add_(0,frames,prediction*weights[:,None,None])
            normalizer.index_add_(0,frames,weights)
    if total is None or (normalizer<=0).any():raise ValueError("Overlapping inference left uncovered frames.")
    return total/normalizer[:,None,None]


def court_zone(parent):
    corners=np.asarray(parent["config"]["calibration"]["corners_px"],dtype=float)
    # Raise the target side of the view, keeping the left neighboring courts cut off.
    left_join=corners[0]+.65*(corners[3]-corners[0])
    polygon=[corners[0].tolist(),corners[1].tolist(),corners[2].tolist(),
             [float(corners[2,0]),0.],[float(left_join[0]),0.],left_join.tolist()]
    if parent['config'].get('shuttle_candidates'):
        # Near-side overheads can travel left of the old arbitrary vertical edge.
        polygon=[corners[0].tolist(),corners[1].tolist(),corners[2].tolist(),
                 [float(corners[2,0]),0.],[float(corners[0,0]),0.]]
        if parent['config'].get('shuttle_association'):
            upper_y=.35*float(corners[2:,1].min())
            polygon=[corners[0].tolist(),corners[1].tolist(),corners[2].tolist(),
                     [float(corners[2,0]),0.],[float(corners[0,0]),0.],
                     [float(corners[0,0]),upper_y],[float(left_join[0]),upper_y],left_join.tolist()]
    return {"kind":"calibrated_court_with_airspace","corners_px":corners.tolist(),"polygon_px":polygon,
            "homography":parent["homography"],"masked_before_selection":True,
            "limitation":"Raised image airspace can overlap neighboring courts and cannot establish match identity. The associated profile widens the upper strip while retaining the floor footprint. Only floor-footprint image points receive court projections."}


def court_mask(parent, width=512, height=288):
    corners=np.asarray(court_zone(parent)["polygon_px"],np.float32)
    # Test heatmap pixel centers in source coordinates; do not round the court outward.
    x,y=np.meshgrid(np.arange(width)*parent["video"]["width"]/width,np.arange(height)*parent["video"]["height"]/height)
    points=np.stack([x,y],axis=-1).reshape(-1,2)
    return np.array([cv2.pointPolygonTest(corners,tuple(map(float,p)),False)>=0 for p in points],dtype=bool).reshape(height,width)


def project_shuttle(table, parent):
    from .upstream import court
    table=table.copy();points=table[["image_x","image_y"]].to_numpy(float)
    candidates=table.detected.eq(True).to_numpy() & np.isfinite(points).all(axis=1)
    zone=court_zone(parent)
    corners=np.asarray(zone["polygon_px"],np.float32)
    inside=np.zeros(len(table),dtype=bool)
    inside[candidates]=[cv2.pointPolygonTest(corners,tuple(map(float,p)),False)>=0 for p in points[candidates]]
    xy=np.full((len(table),2),np.nan)
    floor=np.zeros(len(table),dtype=bool)
    floor[inside]=[cv2.pointPolygonTest(np.asarray(zone["corners_px"],np.float32),tuple(map(float,p)),False)>=0 for p in points[inside]]
    if floor.any():xy[floor]=court.image_to_court(points[floor],np.asarray(parent["homography"]))
    table["detected"]=inside
    table["zone_reason"]=np.where(inside,"inside_court_zone",np.where(candidates,"outside_court_zone","no_in_zone_response"))
    table["projection_x_m"],table["projection_y_m"]=xy[:,0],xy[:,1]
    table["projection_available"]=floor
    return table


def projection_plot(run, table, max_jump_px=120):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from .upstream import viz
    fig,ax=plt.subplots(figsize=(5,8));viz.mpl_court(ax)
    valid=table.detected.to_numpy(bool)&np.isfinite(table[["projection_x_m","projection_y_m"]]).all(axis=1).to_numpy();image=table[["image_x","image_y"]].to_numpy(float)
    xy=table[["projection_x_m","projection_y_m"]].to_numpy(float)
    # Quality mode samples every second source frame; use real time for gaps.
    clock=table.get('original_source_time_s',table.source_time_s)
    delta=np.diff(clock.to_numpy(float))
    cadence=float(np.median(np.diff(table.source_time_s.to_numpy(float)))) if len(delta) else 0.
    consecutive=valid[:-1]&valid[1:]&(delta>0)&(delta<=cadence*1.5)&(np.linalg.norm(np.diff(image,axis=0),axis=1)<=max_jump_px)
    segments=np.stack([xy[:-1][consecutive],xy[1:][consecutive]],axis=1)
    if len(segments):
        lines=LineCollection(segments,cmap="viridis",linewidths=.8,alpha=.65)
        lines.set_array(table.source_time_s.to_numpy()[1:][consecutive]);ax.add_collection(lines)
        fig.colorbar(lines,ax=ax,label="Source seconds",shrink=.7)
    ax.set_title("Birdie image projection\nNot a physical floor trajectory")
    fig.tight_layout();fig.savefig(run/"shuttle_court_projection.png",dpi=140);plt.close(fig)
    return len(segments)


def reuse_shuttle(source, target):
    """Reuse source-aligned shuttle results when only the player review changes."""
    import shutil
    source,target=Path(source),Path(target)
    if not (source/"shuttle.json").exists() or not (source/"shuttle_status.json").exists():return False
    if json.loads((source/"shuttle_status.json").read_text()).get("state")!="complete":return False
    shuttle=json.loads((source/"shuttle.json").read_text());parent=json.loads((target/"analysis.json").read_text())
    if shuttle.get("quality_mode",False)!=parent["config"].get("shuttle_quality",False):return False
    if shuttle.get('candidate_mode',False)!=parent['config'].get('shuttle_candidates',False):return False
    if not shuttle.get("detection_zone") or shuttle["detection_zone"]!=court_zone(parent):return False
    if (shuttle["source_sha256"]!=parent["video"]["sha256"] or shuttle["start_frame"]!=parent["start_frame"]
            or shuttle["end_frame_exclusive"]!=parent["end_frame_exclusive"]):return False
    for name,digest in shuttle["artifacts_sha256"].items():
        if sha256(source/name)!=digest:raise ValueError(f"Cached shuttle artifact changed: {name}")
    for name in ["shuttle.csv","shuttle_reference_template.csv","shuttle_overlay.mp4","shuttle_court_projection.png","shuttle_zone_preview.png","shuttle_status.json","shuttle_validation.json",
                 "shuttle_reviewed_references.csv","shuttle_reference_labels.csv",*shuttle['artifacts_sha256']]:
        if (source/name).exists():shutil.copy2(source/name,target/name)
    shuttle["inference_reused_from"]=str(source.resolve())
    write_json(target/"shuttle.json",shuttle)
    return True


def reference_template(frames):
    selected = np.asarray(frames)[np.linspace(0, len(frames)-1, min(100,len(frames))).round().astype(int)]
    return pd.DataFrame({"source_frame":selected, "visibility":"unreviewed", "image_x":np.nan, "image_y":np.nan})


def heatmap_candidates(heat, width, height):
    contours,_=cv2.findContours((np.asarray(heat)>.5).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    points=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        points.append({'image_x':int((x+w/2)*width/heat.shape[1]),'image_y':int((y+h/2)*height/heat.shape[0]),
                       'confidence':float(heat[y:y+h,x:x+w].max()),'area':w*h})
    return sorted(points,key=lambda p:p['area'],reverse=True)


def evaluate_shuttle(predictions, labels, template, width, height):
    required={"source_frame","visibility","image_x","image_y"}
    if not required <= set(labels):
        raise ValueError("Shuttle references need source_frame, visibility, image_x and image_y.")
    labels=labels.copy()
    numbers=pd.to_numeric(labels.source_frame,errors="coerce")
    if not np.isfinite(numbers).all() or (numbers%1!=0).any() or numbers.duplicated().any():
        raise ValueError("Reference frames must be distinct finite integers.")
    labels["source_frame"]=numbers.astype(int)
    if set(labels.source_frame)!=set(template.source_frame):
        raise ValueError("Complete the exact frozen shuttle reference sample.")
    if not labels.visibility.isin(["visible","not_visible"]).all():
        raise ValueError("Review every shuttle sample as visible or not_visible; uncertain samples remain unreviewed.")
    visible=labels.visibility.eq("visible")
    xy=labels.loc[visible,["image_x","image_y"]].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    if not np.isfinite(xy).all() or (xy<0).any() or (xy[:,0]>=width).any() or (xy[:,1]>=height).any():
        raise ValueError("Visible shuttle references need finite source-image coordinates.")
    labels.loc[visible,["image_x","image_y"]]=xy
    joined=labels.merge(predictions,on="source_frame",suffixes=("_ref",""),validate="one_to_one")
    if len(joined)!=len(labels):
        raise ValueError("Reference frames are missing from shuttle predictions.")
    actual=joined.visibility.eq("visible")
    detected=joined.detected.eq(True)
    distance=np.hypot(joined.image_x-joined.image_x_ref,joined.image_y-joined.image_y_ref)
    tolerance=16*height/720
    correct=actual & detected & distance.le(tolerance)
    tp=int(correct.sum()); fp=int((detected & ~correct).sum()); fn=int((actual & ~correct).sum())
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    matched=distance[actual & detected]
    median=float(matched.median()) if len(matched) else None
    criteria={"100_fixed_samples":len(labels)>=100,"25_visible_references":int(actual.sum())>=25,
              "25_invisible_references":int((~actual).sum())>=25,"precision_90pct":precision>=.9,
              "recall_85pct":recall>=.85,"median_error_8px_at_720p":median is not None and median<=8*height/720}
    return {"status":"passed" if all(criteria.values()) else "failed","criteria":criteria,
            "sample_frames":len(labels),"precision":precision,"recall":recall,"true_positive":tp,
            "false_positive":fp,"false_negative":fn,"median_error_px":median,"match_tolerance_px":tolerance,
            "limitation":"Manual image-point agreement for this source window only; no 3D position, speed or landing claim."}


def validate_shuttle(run, references):
    run=Path(run);manifest=json.loads((run/"shuttle.json").read_text())
    for name,digest in manifest["artifacts_sha256"].items():
        if sha256(run/name)!=digest:raise ValueError(f"Shuttle artifact changed: {name}")
    source=json.loads((run/"analysis.json").read_text())["video"]
    if sha256(source["path"])!=source["sha256"]:raise ValueError("Source video changed.")
    labels=pd.read_csv(references)
    result=evaluate_shuttle(pd.read_csv(run/"shuttle.csv"),labels,pd.read_csv(run/"shuttle_reference_template.csv"),source["width"],source["height"])
    labels.to_csv(run/"shuttle_reviewed_references.csv",index=False)
    result["references_sha256"]=sha256(run/"shuttle_reviewed_references.csv")
    write_json(run/"shuttle_validation.json",result)
    status=json.loads((run/"shuttle_status.json").read_text())
    status["validation"]=result["status"]
    write_json(run/"shuttle_status.json",status)
    from .coaching import write_coaching_report
    write_coaching_report(run)
    return result


def render_shuttle(run, meta, table, start, end, stride=1):
    from .timeline import source_seconds,source_frame_number,crosses_cut
    from .upstream import viz
    destination=run/"shuttle_overlay.mp4"
    indexed=table.set_index("source_frame").to_dict("index")
    parent=json.loads((run/"analysis.json").read_text()) if (run/"analysis.json").exists() else None
    zone=np.asarray(court_zone(parent)["polygon_px"],np.int32) if parent else None
    mini_height=min(meta["height"]-80,viz.court_size()[1]);mini_width=round(viz.court_size()[0]*mini_height/viz.court_size()[1]) if parent else 0
    canvas_width=meta["width"]+(mini_width+24 if parent else 0)
    base=meta["times"][start];tb=Fraction(1,90000);trail=[];projected_trail=[];previous_frame=start-1
    with av.open(meta["path"]) as source,av.open(str(destination),"w") as output:
        stream=output.add_stream("libx264",rate=(Fraction(meta["rate"])/stride).limit_denominator(1001))
        stream.width=canvas_width+canvas_width%2;stream.height=meta["height"]+meta["height"]%2
        stream.pix_fmt="yuv420p";stream.time_base=stream.codec_context.time_base=tb
        stream.options={"crf":"23","preset":"veryfast","bf":"0"}
        for index,decoded in enumerate(source.decode(video=0)):
            if index<start:continue
            if index>=end:break
            if (index-start)%stride:continue
            if crosses_cut(meta,previous_frame,index):trail=[];projected_trail=[]
            previous_frame=index
            frame=decoded.to_ndarray(format="bgr24");r=indexed.get(index)
            if parent:
                cv2.polylines(frame,[np.asarray(parent["config"]["calibration"]["corners_px"],np.int32)],True,(70,140,85),1,cv2.LINE_AA)
                cv2.polylines(frame,[zone],True,(70,230,100),2,cv2.LINE_AA)
                frame=cv2.copyMakeBorder(frame,0,0,0,mini_width+24,cv2.BORDER_CONSTANT,value=(20,26,22))
                mini=viz.render_court()
            if r and r["detected"]:
                point=(round(r["image_x"]),round(r["image_y"]))
                # Break implausible jumps rather than drawing a cross-court false trail.
                if trail and np.linalg.norm(np.subtract(point,trail[-1]))>120*meta["height"]/720:trail=[];projected_trail=[]
                trail=(trail+[point])[-8:]
                for a,b in zip(trail,trail[1:]):cv2.line(frame,a,b,(0,255,255),2,cv2.LINE_AA)
                cv2.circle(frame,point,8,(0,255,255),2)
                if parent and np.isfinite([r.get("projection_x_m",np.nan),r.get("projection_y_m",np.nan)]).all():
                    projected_trail=(projected_trail+[(r["projection_x_m"],r["projection_y_m"])])[-60:]
                    for a,b in zip(projected_trail,projected_trail[1:]):cv2.line(mini,viz.to_px(*a),viz.to_px(*b),(0,255,255),2,cv2.LINE_AA)
                    viz.draw_point(mini,r["projection_x_m"],r["projection_y_m"],(0,255,255),5,"birdie")
                elif parent:
                    projected_trail=[]
                    cv2.putText(mini,"ABOVE FLOOR VIEW",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
            else:trail=[];projected_trail=[]
            if parent:
                mini=cv2.resize(mini,(mini_width,mini_height));frame[36:36+mini_height,meta["width"]+12:meta["width"]+12+mini_width]=mini
                cv2.putText(frame,"IMAGE PROJECTION",(meta["width"]+8,23),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                cv2.putText(frame,"Not true floor position",(meta["width"]+8,meta["height"]-18),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1)
            cv2.rectangle(frame,(0,meta["height"]-32),(meta["width"],meta["height"]),(20,20,20),-1)
            cv2.putText(frame,f"OBSERVED SHUTTLE | original {source_seconds(meta,index):.3f}s | frame {source_frame_number(meta,index)}",(8,meta["height"]-11),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            if parent and index==start:cv2.imwrite(str(run/"shuttle_zone_preview.png"),frame)
            frame=cv2.copyMakeBorder(frame,0,meta["height"]%2,0,canvas_width%2,cv2.BORDER_CONSTANT)
            encoded=av.VideoFrame.from_ndarray(frame,format="bgr24");encoded.pts=round((meta["times"][index]-base)/tb);encoded.time_base=tb
            for packet in stream.encode(encoded):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    return verify_overlay(destination,meta["times"][start:end:stride])


def track_shuttle(run, device="auto"):
    """Use upstream model, preprocessing and heatmap decoding, without InpaintNet."""
    from .runner import choose_device
    run=Path(run).resolve();parent=json.loads((run/"analysis.json").read_text());meta=parent["video"]
    if sha256(meta["path"])!=meta["sha256"]:raise ValueError("Source video changed.")
    if not (run/"shuttle.json").exists():
        candidate=parent.get("parent_run");seen=set()
        while candidate and candidate not in seen:
            seen.add(candidate)
            if reuse_shuttle(candidate,run):
                from .coaching import write_coaching_report
                write_coaching_report(run)
                print(f"Reused source-aligned shuttle inference from {candidate}\nRESULT={run}",flush=True)
                return run
            candidate=json.loads((Path(candidate)/"analysis.json").read_text()).get("parent_run")
    provenance=json.loads((TRACKNET/"PROVENANCE.json").read_text())
    for name,digest in provenance["files"].items():
        if sha256(TRACKNET/name)!=digest:raise ValueError(f"TrackNet source changed: {name}")
    model_path=MODELS/"TrackNet_best.pt";model_manifest=json.loads((MODELS/"tracknet-manifest.json").read_text())
    if sha256(model_path)!=model_manifest["sha256"]:raise ValueError("TrackNet model checksum differs.")
    quality=parent["config"].get("shuttle_quality",False)
    candidates_mode=parent['config'].get('shuttle_candidates',False)
    stride=max(1,round(meta["fps"]/30)) if quality else 1
    start=parent["start_frame"];end=parent["end_frame_exclusive"];frames=list(range(start,end,stride))
    if (run/"shuttle.json").exists() and json.loads((run/"shuttle_status.json").read_text())["state"]=="complete":
        raise ValueError("This run already has shuttle results. Review the cached outputs instead of re-running inference.")
    template_path=run/"shuttle_reference_template.csv"
    reference_template(frames).to_csv(template_path,index=False)
    write_json(run/"shuttle_validation.json",{"status":"pending","reason":"Complete the frozen source-only shuttle references before measuring accuracy."})
    started=time.perf_counter();dev=choose_device(device)
    write_json(run/"shuttle_status.json",{"state":"running","stage":"loading model","processed_frames":0,"total_frames":end-start,"device":dev})
    # These generic upstream module names stay confined to the CLI worker process.
    sys.path.insert(0,str(TRACKNET))
    import torch
    from torch.utils.data import DataLoader
    from dataset import Shuttlecock_Trajectory_Dataset
    from utils.general import get_model
    from predict import predict, get_ensemble_weight
    checkpoint=torch.load(model_path,map_location="cpu",weights_only=True)
    seq=checkpoint["param_dict"]["seq_len"];bg=checkpoint["param_dict"]["bg_mode"]
    model=get_model("TrackNet",seq,bg).to(dev);model.load_state_dict(checkpoint["model"]);model.eval()
    zone_mask=torch.from_numpy(court_mask(parent).astype(np.float32))
    # One shared median avoids per-chunk background changes; sample source independently of predictions.
    sampled=np.linspace(start,end-1,min(64,end-start)).round().astype(int)
    background=np.median(np.stack([read_frame(meta["path"],int(f),meta["times"],meta["pts_origin_s"]) for f in sampled]),axis=0).astype(np.uint8)
    parts=[];chunk=seq*30
    for offset in range(0,len(frames),chunk):
        target_frames=frames[offset:offset+chunk];first=target_frames[0];last=target_frames[-1]+stride
        context=frames[max(0,offset-seq+1):min(len(frames),offset+chunk+seq-1)] if quality else target_frames
        part_path=run/f"shuttle_{'candidates' if candidates_mode else 'quality' if quality else 'airspace'}_chunk_{first}_{last}.csv"
        if part_path.exists():
            table=pd.read_csv(part_path)
            if table.source_frame.tolist()!=target_frames:raise ValueError("Incomplete shuttle checkpoint.")
        else:
            images=[]
            with av.open(meta["path"]) as source:
                target=meta["times"][context[0]]+meta["pts_origin_s"];source.seek(max(0,int(target*av.time_base)),backward=True)
                for frame in source.decode(video=0):
                    if frame.pts is None:raise ValueError("Missing source timestamp.")
                    stamp=float(frame.pts*frame.time_base)-meta["pts_origin_s"]
                    expected=meta["times"][context[len(images)]]
                    if stamp<expected-.0001:continue
                    if abs(stamp-expected)>.0001:raise ValueError("Shuttle source frame alignment failed.")
                    images.append(frame.to_ndarray(format="rgb24"))
                    if len(images)==len(context):break
            if len(images)!=len(context):raise ValueError("Source ended before shuttle interval.")
            if quality and len(context)<seq:raise ValueError("Quality mode requires at least eight sampled source frames.")
            dataset=Shuttlecock_Trajectory_Dataset(seq_len=seq,sliding_step=1 if quality else seq,data_mode="heatmap",bg_mode=bg,frame_arr=np.stack(images),median=background,padding=not quality)
            del images
            rows=[]
            batches=DataLoader(dataset,batch_size=4,num_workers=0,shuffle=False)
            if quality:
                heat=weighted_heatmaps(model,batches,len(context),dev,get_ensemble_weight(seq,"weight"))*zone_mask
                indices=torch.tensor([[[0,i] for i in range(len(context))]])
                decoded_batches=[(indices,heat.unsqueeze(0))]
            else:
                def fast_batches():
                    for indices,x in batches:
                        with torch.inference_mode():heat=model(x.float().to(dev)).cpu()
                        yield indices,heat*zone_mask
                decoded_batches=fast_batches()
            selected=set(target_frames)
            for indices,heat in decoded_batches:
                components={}
                if candidates_mode:
                    for ids,h in zip(indices.reshape(-1,2),heat.reshape(-1,*heat.shape[-2:])):
                        components[int(ids[1])]=heatmap_candidates(h.numpy(),meta['width'],meta['height'])
                prediction=predict(indices,y_pred=heat,img_scaler=(meta["width"]/512,meta["height"]/288))
                for f,px,py,visible in zip(prediction["Frame"],prediction["X"],prediction["Y"],prediction["Visibility"]):
                    sf=context[int(f)]
                    if sf not in selected:continue
                    detected=bool(visible) and 0<=px<meta["width"] and 0<=py<meta["height"]
                    rows.append({"source_frame":sf,"source_time_s":meta["times"][sf],"detected":detected,"image_x":px if detected else None,"image_y":py if detected else None,"interpolated":False})
                    if candidates_mode:rows[-1]['candidates']=json.dumps(components[int(f)])
            table=project_shuttle(pd.DataFrame(rows),parent)
            if table.source_frame.tolist()!=target_frames:raise ValueError("TrackNet did not return every sampled source frame.")
            table.to_csv(part_path,index=False)
            del dataset
        parts.append(table)
        done=min(offset+chunk,len(frames))
        write_json(run/"shuttle_status.json",{"state":"running","stage":"tracking","processed_frames":done,"total_frames":len(frames),"device":dev})
        print(f"Shuttle {done}/{len(frames)} frames; elapsed {time.perf_counter()-started:.1f}s",flush=True)
    table=pd.concat(parts,ignore_index=True)
    from .timeline import source_seconds,source_frame_number
    table['original_source_time_s']=[source_seconds(meta,int(f)) for f in table.source_frame]
    table['original_source_frame']=[source_frame_number(meta,int(f)) for f in table.source_frame]
    table['youtube_time_s']=table.original_source_time_s
    table.to_csv(run/"shuttle.csv",index=False)
    projection_plot(run,table,120*meta["height"]/720)
    write_json(run/"shuttle_status.json",{"state":"running","stage":"rendering","processed_frames":len(table)})
    timing=render_shuttle(run,meta,table,start,end,stride)
    write_json(run/"shuttle.json",{"phase":2,"source_sha256":meta["sha256"],"model":model_manifest,"upstream":provenance,
        "device":dev,"mode":"weighted overlapping windows at about 30 fps" if quality else "native-frame nonoverlap; shared 64-frame median; no inpainting","quality_mode":quality,"candidate_mode":candidates_mode,"tracking_stride":stride,"sampling_fps":meta['fps']/stride,"start_frame":start,"end_frame_exclusive":end,
        "detection_zone":court_zone(parent),"projection":"Floor-footprint image points only are transformed with the floor homography. Airspace-only points have no floor projection; no physical floor location is inferred.",
        "frame_count":len(table),"detected_frames":int(table.detected.sum()),"overlay_timing":timing,"runtime_s":time.perf_counter()-started,
        "artifacts_sha256":{name:sha256(run/name) for name in ["shuttle.csv","shuttle_reference_template.csv"]}})
    write_json(run/"shuttle_status.json",{"state":"complete","processed_frames":len(table),"validation":"pending"})
    from .coaching import write_coaching_report
    write_coaching_report(run)
    print(f"RESULT={run}",flush=True)
    return run
