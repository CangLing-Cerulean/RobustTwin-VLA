from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TASKS = tuple(json.loads((PROJECT / "configs/tasks.json").read_text(encoding="utf-8-sig")))
LOCK = json.loads((PROJECT / "configs/upstream-lock.json").read_text(encoding="utf-8-sig"))
SETTINGS = ("demo_clean", "demo_randomized")
EPISODES_PER_TASK = 50


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def valid_name(name: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", name):
        raise ValueError("Run name must be 1-80 letters, digits, underscores or hyphens.")
    return name


def checked_relative(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes its root: {value}")
    return path


def cloud_env(source: Path, gpu_count: int) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env.pop(key, None)
    env.update(
        HIP_VISIBLE_DEVICES=",".join(map(str, range(gpu_count))),
        AITER_TRITON_ONLY="1", FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE",
        ROBOTWIN_DISABLE_CUROBO="1", ROBOTWIN_EE_PLANNER="mplib",
        PYOPENGL_PLATFORM="egl", PYTHONUNBUFFERED="1",
        SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0",
        PYTHONPATH=str(source) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
    )
    return env


def run_logged(command: list[str], cwd: Path, env: dict, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            result = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if result:
        raise RuntimeError(f"Command failed ({result}); see {log}")
