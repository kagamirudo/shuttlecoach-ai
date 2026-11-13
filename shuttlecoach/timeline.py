"""Keep original source clocks when reviewed play is assembled with context."""
from pathlib import Path

import numpy as np
import pandas as pd

from .media import sha256


def attach_timeline(meta, config):
    mapping=config.get('source_frame_map')
    if not mapping:return meta
    path=Path(mapping['path'])
    if sha256(path)!=mapping['sha256']:raise ValueError('Source frame map changed.')
    table=pd.read_csv(path)
    if len(table)!=meta['frame_count'] or not np.array_equal(table.clip_frame,np.arange(meta['frame_count'])):
        raise ValueError('Frame map must cover every assembled video frame exactly once.')
    frames=table.original_source_frame.to_numpy(float);times=table.original_source_time_s.to_numpy(float)
    if not np.isfinite(frames).all() or not np.isfinite(times).all() or (frames%1!=0).any() or (frames<0).any() or (times<0).any() or (np.diff(frames)<=0).any() or (np.diff(times)<=0).any():
        raise ValueError('Original frame numbers and times must be finite and strictly increasing.')
    original=config['original_source']
    if (original.get('width'),original.get('height'))!=(meta.get('width'),meta.get('height')):
        raise ValueError('Mapped play must preserve the original frame geometry.')
    if sha256(original['path'])!=original['sha256']:raise ValueError('Original match changed.')
    if times[-1]>=original['duration_s']:raise ValueError('Frame map exceeds original match duration.')
    meta=meta.copy();meta.update(original_source=original,original_times=times.tolist(),original_frames=frames.astype(int).tolist(),
                                cut_frames=(np.flatnonzero(np.diff(frames)>1)+1).tolist())
    return meta


def source_seconds(meta, frame):
    return meta.get('original_times',meta['times'])[frame]


def source_frame_number(meta, frame):
    return meta['original_frames'][frame] if 'original_frames' in meta else frame


def crosses_cut(meta, previous, current):
    return any(previous<c<=current for c in meta.get('cut_frames',[]))
