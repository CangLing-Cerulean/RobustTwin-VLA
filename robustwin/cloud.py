from __future__ import annotations

import copy
import json
import math
import platform
import shutil
import subprocess
from pathlib import Path

import yaml

from .common import LOCK, PROJECT, TASKS, canonical_hash, cloud_env, now, read_json, run_logged, sha256, valid_name, write_json
from .data import audit_data

GIB = 1024 ** 3
DCP_GIB = 70
EXPORT_AND_LOG_GIB = 20
EXPERIMENT = "experiments/lingbot_vla_v2_6b_robotwin"


def paths(root: Path, runtime: Path) -> dict[str, Path]:
    experiment = root / EXPERIMENT
    return {"root": root, "experiment": experiment, "source": experiment / "source/lingbot-vla-v2",
            "models": experiment / "models", "runtime": runtime, "prepared": runtime / "prepared"}


def plan(gpus: int, steps: int, save_every: int, episodes: int) -> dict:
    if gpus not in (4, 8) or min(steps, save_every, episodes) <= 0:
        raise ValueError("Use 4 or 8 GPUs and positive steps/save interval/episodes.")
    # AMD measured capacity-test step times and measured 100-task x 10-episode runs.
    step_seconds = {4: 41.392, 8: 24.682}[gpus]
    evaluation_hours = {4: 18 + 55 / 60 + 48 / 3600, 8: 15 + 12 / 60 + 21 / 3600}[gpus] * episodes / 10
    train_hours = steps * step_seconds / 3600
    return {"gpus": gpus, "micro_batch_size": 16, "gradient_accumulation_steps": 256 // (gpus * 16),
            "global_batch_size": 256, "steps": steps, "save_every": save_every,
            "training_hours_estimate": round(train_hours, 2), "self_eval_hours_estimate": round(evaluation_hours, 2),
            "training_credits_estimate": round(train_hours * gpus, 1),
            "self_eval_credits_estimate": round(evaluation_hours * gpus, 1),
            "credits_with_20_credit_setup_reserve": round((train_hours + evaluation_hours) * gpus + 20, 1),
            "additional_free_disk_gib": math.ceil(steps / save_every) * DCP_GIB + EXPORT_AND_LOG_GIB,
            "checkpoint_policy": "No automatic deletion. One end-of-run DCP by default; shorter save intervals require more disk.",
            "evaluation": {"tasks": 50, "settings": 2, "episodes_per_task_setting": episodes,
                           "total_episodes": 100 * episodes, "type": "self_evaluation"},
            "estimate_limit": "Planning estimate from AMD reference measurements, not a runtime or credit guarantee. Time data differ between official documents; benchmark the allocated instance first."}


def verify_contract(p: dict) -> dict:
    checked = {}
    for relative, expected in LOCK["reference_files"].items():
        file = p["experiment"] / relative
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"AMD image interface changed or is missing: {file}. Recheck the pinned reference before training.")
        checked[relative] = expected
    for relative, key in (("tasks/vla/train_lingbotvla.py", "trainer_sha256"), ("scripts/compute_norm_stats.py", "norm_script_sha256")):
        file = p["source"] / relative
        if not file.is_file() or sha256(file) != LOCK[key]:
            raise ValueError(f"Pinned LingBot source mismatch: {file}")
    for relative, expected in LOCK["source_files"].items():
        file = p["source"] / relative
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"Pinned data/inference source mismatch: {file}")
    for directory, key in ((p["root"], "robotwin_revision"), (p["root"] / "XPolicyLab", "xpolicylab_revision"), (p["source"], "lingbot_revision")):
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if actual != LOCK[key]:
            raise ValueError(f"Revision mismatch at {directory}: {actual}; expected {LOCK[key]}")
    tasks = yaml.safe_load((p["root"] / "env_cfg/eval/all_tasks.yml").read_text(encoding="utf-8"))["tasks"]
    if len(tasks) != 50 or set(tasks) != set(TASKS):
        raise ValueError("Cloud task list differs from the 50 pinned competition tasks.")
    return checked


def doctor(root: Path, runtime: Path, python: str, gpus: int, required_disk: int = 0) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("GPU execution requires the Linux AMD robotwin image. Local Windows supports plan, report, bundle checks and unit tests.")
    p = paths(root, runtime)
    contracts = verify_contract(p)
    probe = """import json, torch, flash_attn, sapien, mplib, pyarrow, lerobot
print(json.dumps({'torch': torch.__version__, 'hip': torch.version.hip,
'flash_attn': flash_attn.__version__, 'gpus': [{'name': torch.cuda.get_device_name(i),
'memory_gib': round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)}
for i in range(torch.cuda.device_count())]}))"""
    result = subprocess.run([python, "-c", probe], text=True, capture_output=True, check=True,
                            env=cloud_env(p["source"], gpus))
    probe_result = json.loads(result.stdout.strip().splitlines()[-1])
    if not probe_result["hip"] or len(probe_result["gpus"]) < gpus:
        raise ValueError(f"Expected {gpus} AMD GPUs; got {probe_result}")
    if not probe_result["torch"].startswith("2.9.1") or not probe_result["hip"].startswith("7.2") or probe_result["flash_attn"] != "2.8.4":
        raise ValueError(f"Unexpected runtime versions for the pinned AMD image: {probe_result}")
    if any(gpu["memory_gib"] < 47 for gpu in probe_result["gpus"][:gpus]):
        raise ValueError("This recipe targets W7900 48GB GPUs; smaller GPUs need a separately validated configuration.")
    model_root = p["models"]
    base = model_root / "robbyant_lingbot-vla-v2-6b"
    for required in (base / "model.safetensors.index.json", base / "config.json", base / "depth/model.pt",
                     base / "dino_video/teacher_step_10000.pth", base / "dino_video/config.yaml",
                     model_root / "moge-2-vitb-normal/model.pt", model_root / "Qwen3-VL-4B-Instruct-config-tokenizer/config.json"):
        if not required.is_file():
            raise ValueError(f"Missing image model asset: {required}; verify the Devzone mount.")
    from .artifacts import weight_files
    weight_files(base)
    existing = runtime
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free / GIB
    if free < required_disk:
        raise ValueError(f"Need at least {required_disk} GiB additional free disk at {runtime}; available {free:.1f} GiB. No checkpoint is deleted automatically.")
    return {"checked_at": now(), "environment": probe_result, "free_disk_gib": round(free, 1),
            "source_contracts": contracts, "model_origin": LOCK["base_model"],
            "model_revision_expected": LOCK["base_model_revision"],
            "model_provenance_limit": "Local model files are checked for completeness; revision comes from the pinned AMD image, not a Hugging Face remote signature."}


def prepare(root: Path, runtime: Path, python: str, gpus: int) -> dict:
    p = paths(root, runtime)
    audit = audit_data(root, root / "data/robotwin_demo_clean_joint_v30.txt", deep=True)
    target = p["prepared"]
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"Prepared directory already contains data: {target}; use a new runtime to recompute.")
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / "train_clean.txt"
    manifest.write_text("".join(f"robotwin {audit['tasks'][task]['path']}\n" for task in TASKS), encoding="utf-8")
    # MultiVLADataset reads norm_stats from the robot YAML, not data.norm_stats_file.
    # Bind both training and inference to the same clean-only statistics explicitly.
    robot_dir = target / "robot_configs"
    robot_dir.mkdir()
    robot_config = yaml.safe_load((p["source"] / "configs/robot_configs/robotwin.yaml").read_text(encoding="utf-8"))
    robot_config["norm_stats"] = str(target / "norm_stats.json")
    robot_file = robot_dir / "robotwin.yaml"
    robot_file.write_text(yaml.safe_dump(robot_config, sort_keys=False), encoding="utf-8")
    config = {"data": {"datasets_type": "vla", "data_name": "multi", "robot_name": "robotwin",
                        "train_path": str(manifest), "robot_config_root": str(p["source"] / "configs/robot_configs"),
                        "norm_path": str(target / "norm_stats.json"), "data_ratio_for_norm_compute": 1.0,
                        "norm_merge_chunk_dim": True, "num_workers": 4, "prompt_type": "global"},
              "train": {"micro_batch_size": 48, "chunk_size": 50, "output_dir": "", "max_steps": 100000000000000}}
    config_path = target / "norm-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [python, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={gpus}",
               str(p["source"] / "scripts/compute_norm_stats.py"), str(config_path)]
    write_json(target / "data-audit.json", audit)
    run_logged(command, p["source"], cloud_env(p["source"], gpus), target / "norm.log")
    validate_norm(target / "norm_stats.json")
    audit["norm_sha256"] = sha256(target / "norm_stats.json")
    audit["manifest_sha256"] = sha256(manifest)
    audit["robot_config_sha256"] = sha256(robot_file)
    audit["completed_at"] = now()
    write_json(target / "prepared.json", audit)
    return audit


def validate_norm(path: Path) -> None:
    stats = read_json(path)["norm_stats"]
    for key in ("action.arm.position", "action.effector.position", "observation.state.arm.position", "observation.state.effector.position"):
        row = stats[key]
        dimensions = 12 if key.endswith("arm.position") else 2
        for field in ("mean", "std", "q01", "q99"):
            values = row[field]
            if len(values) != dimensions or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
                raise ValueError(f"Invalid normalization values: {key}.{field}")
        if any(low > high for low, high in zip(row["q01"], row["q99"])) or any(x < 0 for x in row["std"]):
            raise ValueError(f"Invalid normalization ranges: {key}")


def teacher_config(models: Path) -> dict:
    base = models / "robbyant_lingbot-vla-v2-6b"
    return {
        "mode": "query", "num_task_tokens": 8, "depth_loss_weight": 0.004,
        "future_depth_loss_weight": 0.004, "use_future_video": True,
        "llm": {"dim_out": 2560, "image_token_size": 8, "image_input_size": 224},
        "depth": {"model_type": "MoRGBD", "moge_path": str(models / "moge-2-vitb-normal/model.pt"),
                  "morgbd_path": str(base / "depth/model.pt"), "num_layers": 1, "num_heads": 4, "dim_head": 32,
                  "ff_mult": 1, "num_backbone_tokens": 256, "token_size": 16, "dim_out": 1024, "input_size": 224,
                  "use_future_depth": True, "block_future_depth_to_action": True, "detach_future_image_feats": True},
        "video": {"ckpt_path": str(base / "dino_video/teacher_step_10000.pth"), "config_path": str(base / "dino_video/config.yaml"),
                  "attention_mode": "flex_block_causal", "input_size": 256, "block_suffix_to_future_video": True,
                  "block_warmup_steps": 0, "block_warmup_gradual": False, "share_future_depth_query": True,
                  "use_shared_future_task_proj": True, "use_current_shared_task_proj": True, "shared_query_head_type": "resampler",
                  "num_future_frames": 1, "use_warmup_frame": True, "effective_fps": 1.0, "n_blocks": 1, "cls_pool": "last",
                  "head_type": "resampler", "detach_image_feats": True, "num_layers": 1, "num_heads": 4,
                  "dim_head": 32, "ff_mult": 1, "num_backbone_tokens": 256, "dim_out": 1024, "target_type": "absolute",
                  "future_video_loss_weight": 0.004, "use_smooth_l1_loss": False, "use_mse_loss": True, "mse_loss_weight": 1.0,
                  "use_patch_loss": True, "use_current_patch_loss": True, "use_cosine_loss": False, "cosine_loss_weight": 0.2,
                  "use_cls_loss": False, "cls_loss_type": "mse", "cls_loss_weight": 0.2, "log_max_samples": 32, "log_scale": 16},
        "visual_steps": 5000,
    }


def training_config(base: dict, p: dict, run_name: str, recipe: str, gpus: int, steps: int, save_every: int) -> dict:
    plan(gpus, steps, save_every, 2)
    valid_name(run_name)
    if recipe not in ("baseline", "robust"):
        raise ValueError("Recipe must be baseline or robust.")
    config = copy.deepcopy(base)
    config["data"].update(train_path=str(p["prepared"] / "train_clean.txt"), data_name="multi",
                          norm_stats_file=str(p["prepared"] / "norm_stats.json"),
                          robot_config_root=str(p["prepared"] / "robot_configs"),
                          use_future_image=True, image_augment=recipe == "robust", num_workers=4)
    model = p["models"] / "robbyant_lingbot-vla-v2-6b"
    config["model"].update(model_path=str(model), config_path=str(model), post_training=True,
                           tokenizer_path=str(p["models"] / "Qwen3-VL-4B-Instruct-config-tokenizer"))
    config["train"].update(output_dir=str(p["runtime"] / "training" / run_name), use_lora=False,
                           train_expert_only=False, freeze_vision_encoder=False, freeze_vit=False,
                           optimizer="adamw", data_parallel_mode="fsdp2", data_parallel_replicate_size=1,
                           data_parallel_shard_size=gpus, micro_batch_size=16,
                           gradient_accumulation_steps=256 // (gpus * 16), global_batch_size=256,
                           enable_gradient_checkpointing=True, enable_full_shard=True,
                           enable_resume=False, load_checkpoint_path=None, allow_partial_checkpoint=False,
                           max_steps=steps, save_steps=save_every, save_epochs=0,
                           save_hf_weights=False, async_save_hf_weights=False,
                           seed=42, align_params=teacher_config(p["models"]))
    return config


def train(root: Path, runtime: Path, python: str, gpus: int, run_name: str, recipe: str,
          steps: int, save_every: int, resume_step: int | None = None, environment_report: dict | None = None) -> dict:
    p = paths(root, runtime)
    prepared = read_json(p["prepared"] / "prepared.json")
    if prepared["norm_sha256"] != sha256(p["prepared"] / "norm_stats.json") or prepared["manifest_sha256"] != sha256(p["prepared"] / "train_clean.txt"):
        raise ValueError("Prepared data/normalization changed after audit.")
    if prepared["robot_config_sha256"] != sha256(p["prepared"] / "robot_configs/robotwin.yaml"):
        raise ValueError("Prepared robot mapping changed after audit.")
    current = audit_data(root, p["prepared"] / "train_clean.txt")
    if current["fingerprint"] != prepared["fingerprint"]:
        raise ValueError("Dataset metadata changed after preparation.")
    base = yaml.safe_load((p["experiment"] / "training/lingbotvla_cli.yaml").read_text(encoding="utf-8"))
    config = training_config(base, p, run_name, recipe, gpus, steps, save_every)
    run_dir = Path(config["train"]["output_dir"])
    identity = canonical_hash({"config": config, "data": prepared["fingerprint"], "norm": prepared["norm_sha256"],
                               "robot_config": prepared["robot_config_sha256"], "lock": LOCK})
    if resume_step is not None:
        record = read_json(run_dir / "robustwin-run.json")
        if record["identity"] != identity or not 0 < resume_step < steps:
            raise ValueError("Resume must use the same configuration/data, and an intermediate checkpoint. A new schedule requires a new run.")
        checkpoint = run_dir / "checkpoints" / f"global_step_{resume_step}"
        if not checkpoint.is_dir() or not any(checkpoint.rglob(".metadata")):
            raise ValueError(f"No complete DCP metadata found: {checkpoint}")
        config["train"]["load_checkpoint_path"] = str(checkpoint)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"Training output is not empty: {run_dir}; use a new run name or explicit --resume-step.")
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {"identity": identity, "created_at": now(), "recipe": recipe, "steps": steps,
                  "gpus": gpus, "base_model": LOCK["base_model"], "data_fingerprint": prepared["fingerprint"],
                  "norm_sha256": prepared["norm_sha256"], "status": "prepared", "config": config, "upstream": LOCK,
                  "environment": environment_report}
    config_path = run_dir / "robustwin-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    shutil.copy2(p["prepared"] / "norm_stats.json", run_dir / "robustwin-norm-stats.json")
    shutil.copy2(p["prepared"] / "robot_configs/robotwin.yaml", run_dir / "robustwin-robotwin.yaml")
    shutil.copy2(p["prepared"] / "prepared.json", run_dir / "data-provenance.json")
    record.update(status="running", last_started_at=now(), resume_step=resume_step)
    write_json(run_dir / "robustwin-run.json", record)
    env = cloud_env(p["source"], gpus)
    env["PYTHONPATH"] = str(PROJECT) + ":" + env["PYTHONPATH"]
    env["ROBUSTWIN_TRAINER_PATH"] = str(p["source"] / "tasks/vla/train_lingbotvla.py")
    command = [python, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={gpus}",
               "-m", "robustwin.training_entry", str(config_path)]
    try:
        run_logged(command, p["source"], env, run_dir / "train.log")
        checkpoint = run_dir / "checkpoints" / f"global_step_{steps}"
        if not checkpoint.is_dir() or not any(checkpoint.rglob(".metadata")):
            raise ValueError("Training exited without the expected final DCP checkpoint.")
        record.update(status="trained", completed_at=now(), checkpoint=str(checkpoint))
    except BaseException:
        record.update(status="interrupted_or_failed", stopped_at=now())
        write_json(run_dir / "robustwin-run.json", record)
        raise
    write_json(run_dir / "robustwin-run.json", record)
    return record
