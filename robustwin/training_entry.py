"""Run the pinned AMD trainer with strict resume failure semantics.

The upstream trainer catches a failed checkpoint load and starts from scratch.
Replace exactly that terminal branch in memory, leaving the source tree intact.
"""
from __future__ import annotations

import os
from pathlib import Path

from .common import LOCK, sha256

ORIGINAL = '            logger.info_rank0("Starting training from scratch. No valid checkpoint could be loaded.")'
STRICT = '            raise RuntimeError("RobustTwin: requested checkpoint could not be loaded; refusing to restart training.") from last_err'


def strict_resume_source(source: str) -> str:
    if source.count(ORIGINAL) != 1:
        raise ValueError("Pinned trainer resume branch changed; review the upstream code first.")
    return source.replace(ORIGINAL, STRICT)


if __name__ == "__main__":
    trainer_path = Path(os.environ["ROBUSTWIN_TRAINER_PATH"])
    if sha256(trainer_path) != LOCK["trainer_sha256"]:
        raise ValueError("Training source hash mismatch.")
    trainer_source = strict_resume_source(trainer_path.read_text(encoding="utf-8"))
    __file__ = str(trainer_path)
    exec(compile(trainer_source, str(trainer_path), "exec"), globals())
