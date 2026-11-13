"""Bind an unmodified, pinned BadmintonAI detector to an isolated run.

The upstream detector writes through its config/db modules. Only these IO bindings
are scoped here; detection, ByteTrack selection and ground-point logic are upstream.
Run inference in a separate worker process, never concurrently inside Streamlit.
"""
from contextlib import contextmanager
from functools import partial
import sys

from .paths import VENDOR

sys.path.insert(0, str(VENDOR / "src"))
from badminton import config, court, db, detect, viz  # noqa: E402
from badminton.render_overlay import draw_skeleton  # noqa: E402


@contextmanager
def bind_run(video, homography, database):
    old_get, old_root, old_transaction = config.get_match, config.REPO_ROOT, db.transaction
    config.REPO_ROOT = video.parent
    config.get_match = lambda match_id: {"video_path": video.name, "homography": homography}
    db.transaction = partial(old_transaction, database)
    try:
        db.init_db(database)
        with old_transaction(database) as con:
            con.execute("INSERT INTO matches(match_id,discipline) VALUES ('clip','singles') ON CONFLICT DO NOTHING")
        yield
    finally:
        config.get_match, config.REPO_ROOT, db.transaction = old_get, old_root, old_transaction
