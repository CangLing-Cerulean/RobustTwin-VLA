from __future__ import annotations

import json
import shutil
import struct
import zipfile
from pathlib import Path

import yaml

from .common import LOCK, PROJECT, canonical_hash, checked_relative, cloud_env, now, read_json, run_logged, sha256, valid_name, write_json
from .cloud import paths, validate_norm
from .results import write_report


def weight_files(model: Path) -> list[Path]:
    if not (model / "config.json").is_file():
        raise ValueError(f"Missing model config: {model}")
    index = model / "model.safetensors.index.json"
    if index.is_file():
        mapping = read_json(index).get("weight_map", {})
        if not mapping:
            raise ValueError("Empty model weight index.")
        files = [checked_relative(model, item) for item in sorted(set(mapping.values()))]
    else:
        files = [model / "model.safetensors"]
    for file in files:
        if not file.is_file() or file.stat().st_size < 10:
            raise ValueError(f"Missing or empty weight shard: {file}")
    return files


def check_safetensors(path: Path) -> None:
    """Check framing and offsets without allocating model tensors."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length <= 0 or length > min(size - 8, 100_000_000):
            raise ValueError(f"Invalid safetensors header: {path}")
        header = json.loads(stream.read(length))
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    if not tensors:
        raise ValueError(f"No tensors in {path}")
    for tensor in tensors:
        start, end = tensor["data_offsets"]
        if not 0 <= start <= end <= size - 8 - length:
            raise ValueError(f"Invalid tensor offset in {path}")


def model_manifest(model: Path) -> dict:
    files = {}
    for file in sorted(model.rglob("*")):
        if file.is_symlink():
            raise ValueError(f"Export bundle must contain files rather than symlinks: {file}")
        if file.is_file() and file.name != "bundle-manifest.json":
            files[file.relative_to(model).as_posix()] = {"bytes": file.stat().st_size, "sha256": sha256(file)}
    return {"schema_version": 1, "files": files, "identity": canonical_hash(files)}


def verify_model(model: Path) -> str:
    for file in weight_files(model):
        check_safetensors(file)
    for required in ("lingbotvla_cli.yaml", "norm_stats.json", "robot_configs/robotwin.yaml", "training-record.json"):
        if not (model / required).is_file():
            raise ValueError(f"Missing portable model asset: {required}")
    validate_norm(model / "norm_stats.json")
    expected = read_json(model / "bundle-manifest.json")
    actual = model_manifest(model)
    if actual != expected:
        raise ValueError("Model bundle changed after export; manifest hashes do not match.")
    record = read_json(model / "training-record.json")
    if record.get("status") != "trained" or record.get("base_model") != LOCK["base_model"]:
        raise ValueError("Model lacks a completed training record from the required base model.")
    return actual["identity"]


def export_model(root: Path, runtime: Path, python: str, run_name: str) -> dict:
    p = paths(root, runtime)
    run = runtime / "training" / valid_name(run_name)
    record = read_json(run / "robustwin-run.json")
    if record["status"] != "trained":
        raise ValueError("Only a completed training run can be exported.")
    target = runtime / "models" / run_name
    if target.exists():
        raise ValueError(f"Export destination already exists: {target}")
    command = [python, str(p["experiment"] / "scripts/convert_full_sft_dcp.py"),
               "--checkpoint", record["checkpoint"], "--training-output", str(run), "--output", str(target)]
    run_logged(command, p["root"], cloud_env(p["source"], record["gpus"]), run / "export.log")
    (target / "robot_configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "robustwin-norm-stats.json", target / "norm_stats.json")
    shutil.copy2(run / "robustwin-robotwin.yaml", target / "robot_configs/robotwin.yaml")
    shutil.copy2(run / "robustwin-config.yaml", target / "lingbotvla_cli.yaml")
    shutil.copy2(run / "train.log", target / "training.log")
    shutil.copy2(run / "data-provenance.json", target / "data-provenance.json")
    write_json(target / "training-record.json", record)
    write_json(target / "bundle-manifest.json", model_manifest(target))
    identity = verify_model(target)
    return {"model": str(target), "identity": identity, "status": "exported_not_yet_evaluated"}


def evaluate(root: Path, runtime: Path, python: str, gpus: int, run_name: str,
             model: Path, episodes: int, resume: bool = False) -> dict:
    if episodes <= 0:
        raise ValueError("Episodes must be positive.")
    p = paths(root, runtime)
    model = model.resolve()
    model_id = verify_model(model)
    name = valid_name(run_name)
    evaluation_runtime = runtime / "evaluation"
    run = evaluation_runtime / "outputs" / name
    protocol = {"model_identity": model_id, "episodes": episodes, "gpus": gpus, "settings": ["demo_clean", "demo_randomized"],
                "seed": 0, "planner": "mplib", "expert_check": True, "accept_expert_info_on_failure": True,
                "precision": "bf16", "use_fp32": False, "use_length": 25, "video": False,
                "benchmark_script_sha256": LOCK["reference_files"]["scripts/run_clean_benchmark.py"]}
    identity = canonical_hash(protocol)
    metadata = run / "robustwin-eval.json"
    if run.exists() and any(run.iterdir()):
        if not resume or not metadata.is_file() or read_json(metadata)["identity"] != identity:
            raise ValueError("Evaluation directory belongs to a different run, model or protocol; use a new name.")
    else:
        if resume:
            raise ValueError("No recorded evaluation to resume.")
        run.mkdir(parents=True, exist_ok=True)
        write_json(metadata, {**protocol, "identity": identity, "created_at": now(), "official_score": None})
    # The export is relocatable: bind local absolute paths at evaluation time.
    deployment = yaml.safe_load((model / "lingbotvla_cli.yaml").read_text(encoding="utf-8"))
    deployment["data"].update(norm_stats_file=str(model / "norm_stats.json"),
                              robot_config_root=str(model / "robot_configs"), image_augment=False)
    deployment["model"]["model_path"] = str(model)
    deployment_path = run / "deployment.yaml"
    deployment_path.write_text(yaml.safe_dump(deployment, sort_keys=False), encoding="utf-8")
    env = cloud_env(p["source"], gpus)
    env["LINGBOTVLA_TRAINING_CONFIG"] = str(deployment_path)
    command = [python, str(p["experiment"] / "scripts/run_clean_benchmark.py"), "--gpu-count", str(gpus),
               "--episodes", str(episodes), "--task-config", "both", "--expert-check", "--accept-expert-info-on-failure",
               "--no-video", "--model-path", str(model), "--run-name", name, "--runtime-dir", str(evaluation_runtime),
               "--robotwin-root", str(root), "--python", python, "--resume"]
    run_logged(command, root, env, run / "benchmark.log")
    return write_report(run, episodes)


def bundle(model: Path, evaluation: Path, output: Path) -> dict:
    identity = verify_model(model)
    metadata = read_json(evaluation / "robustwin-eval.json")
    report = write_report(evaluation, metadata["episodes"])
    if metadata["model_identity"] != identity:
        raise ValueError("Evaluation results belong to a different model.")
    if output.exists():
        raise ValueError(f"Bundle already exists: {output}")
    files = []
    for directory in ("robustwin", "configs", "docs", "tests"):
        files.extend((file, f"code/{file.relative_to(PROJECT).as_posix()}") for file in (PROJECT / directory).rglob("*")
                     if file.is_file() and "__pycache__" not in file.parts)
    for name in ("README.md", "requirements.txt", "requirements-test.txt", "THIRD_PARTY_NOTICES.md"):
        files.append((PROJECT / name, "code/" + name))
    files.extend((file, f"model/{file.relative_to(model).as_posix()}") for file in model.rglob("*") if file.is_file())
    for name in ("robustwin-eval.json", "report.json", "results.csv", "report.html", "deployment.yaml", "benchmark.log"):
        if (evaluation / name).is_file():
            files.append((evaluation / name, "evaluation/" + name))
    for directory in ("logs", "done", "episode_info"):
        files.extend((file, f"evaluation/{file.relative_to(evaluation).as_posix()}")
                     for file in (evaluation / directory).rglob("*") if file.is_file())
    for file, _ in files:
        if file.is_symlink():
            raise ValueError(f"Refusing symlink in submission: {file}")
    output.parent.mkdir(parents=True, exist_ok=True)
    size = sum(file.stat().st_size for file, _ in files)
    if shutil.disk_usage(output.parent).free < size * 1.02:
        raise ValueError("Not enough free disk for the submission archive.")
    manifest = {"generated_at": now(), "model_identity": identity, "submission_type": "candidate_materials",
                "official_score": None, "self_eval_episodes": metadata["episodes"],
                "covers_100_episode_protocol": report["covers_100_episode_protocol"],
                "packaging_limit": "Engineering bundle. Reconcile Tianchi's final field names, archive layout and upload limits before submitting.",
                "files": {name: {"bytes": file.stat().st_size, "sha256": sha256(file)} for file, name in files}}
    temp = output.with_suffix(output.suffix + ".partial")
    if temp.exists():
        raise ValueError(f"Previous partial bundle exists: {temp}; inspect it before retrying.")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for file, name in files:
            archive.write(file, name)
        archive.writestr("submission-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    temp.replace(output)
    return {"archive": str(output), "sha256": sha256(output), "bytes": output.stat().st_size,
            "status": "candidate_materials_not_uploaded", "official_score": None}
