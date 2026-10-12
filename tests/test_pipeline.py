from __future__ import annotations

import contextlib
import copy
import io
import importlib.util
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import yaml

from robustwin import artifacts, cloud
from robustwin.__main__ import main
from robustwin.common import LOCK, PROJECT, SETTINGS, TASKS, canonical_hash, sha256, write_json
from robustwin.data import CAMERAS, audit_data, audit_rows
from robustwin.results import collect, parse_result, wilson, write_report
from robustwin.training_entry import ORIGINAL, strict_resume_source


def dataset_fixture(root: Path) -> Path:
    """Synthetic metadata fixtures only; these are NOT robot training data."""
    lines = []
    for task in TASKS:
        data = root / "data/lerobot" / f"{task}_joint_v30"
        (root / "data/demo_clean" / task).mkdir(parents=True)
        (data / "data").mkdir(parents=True)
        (data / "data/chunk.parquet").write_bytes(b"metadata-only unit test placeholder")
        features = {key: {"shape": [14], "dtype": "float32"} for key in ("observation.state", "action")}
        for camera in CAMERAS:
            features[camera] = {"shape": [256, 256, 3], "dtype": "video"}
            (data / "videos" / camera).mkdir(parents=True)
            (data / "videos" / camera / "part.mp4").write_bytes(b"metadata-only unit test placeholder")
        write_json(data / "meta/info.json", {"codebase_version": "v3.0", "total_episodes": 50,
                                             "total_frames": 100, "features": features})
        lines.append(f"robotwin {data.resolve()}")
    manifest = root / "manifest.txt"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def norm_fixture() -> dict:
    return {"norm_stats": {key: {"mean": [0.0] * count, "std": [1.0] * count,
                                "q01": [-1.0] * count, "q99": [1.0] * count}
                           for key, count in (("action.arm.position", 12), ("action.effector.position", 2),
                                              ("observation.state.arm.position", 12), ("observation.state.effector.position", 2))}}


def model_fixture(model: Path) -> str:
    """Tiny safetensors framing fixture, never a claimed trained model."""
    model.mkdir(parents=True)
    header = json.dumps({"test_tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (model / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1))
    write_json(model / "config.json", {"unit_test_only": True})
    write_json(model / "norm_stats.json", norm_fixture())
    write_json(model / "training-record.json", {"status": "trained", "base_model": LOCK["base_model"], "unit_test_only": True})
    (model / "robot_configs").mkdir()
    (model / "robot_configs/robotwin.yaml").write_text("states: []\n", encoding="utf-8")
    (model / "lingbotvla_cli.yaml").write_text("data: {}\nmodel: {}\n", encoding="utf-8")
    manifest = artifacts.model_manifest(model)
    write_json(model / "bundle-manifest.json", manifest)
    return manifest["identity"]


def result_fixture(run: Path, episodes=10, model_identity="unit-test-only") -> None:
    (run / "logs").mkdir(parents=True)
    (run / "done").mkdir()
    for setting in SETTINGS:
        successes = episodes if setting == "demo_clean" else 0
        for task in TASKS:
            key = f"{setting}__{task}"
            (run / "logs" / f"eval_{key}.log").write_text(
                f"UNIT TEST FIXTURE\nFinal success rate: {successes}/{episodes} = {successes / episodes * 100:.2f}%\n", encoding="utf-8")
            (run / "done" / f"{key}.done").touch()
    write_json(run / "robustwin-eval.json", {"episodes": episodes, "model_identity": model_identity, "unit_test_only": True})


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = dataset_fixture(self.root)

    def test_full_clean_manifest(self):
        report = audit_data(self.root, self.manifest)
        self.assertEqual(report["episode_count"], 2500)
        self.assertEqual(report["task_count"], 50)
        self.assertEqual(report["scan"], "metadata_and_file_presence")

    def test_duplicates_and_missing_tasks_are_rejected(self):
        lines = self.manifest.read_text(encoding="utf-8").splitlines()
        self.manifest.write_text("\n".join(lines[:-1] + [lines[0]]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_data(self.root, self.manifest)
        self.manifest.write_text("\n".join(lines[:-1]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing tasks"):
            audit_data(self.root, self.manifest)

    def test_randomized_dataset_name_is_rejected(self):
        text = self.manifest.read_text(encoding="utf-8").replace("adjust_bottle_joint_v30", "adjust_bottle_randomized_joint_v30")
        self.manifest.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unexpected task"):
            audit_data(self.root, self.manifest)

    def test_wrong_episode_count_is_rejected(self):
        file = self.root / "data/lerobot/adjust_bottle_joint_v30/meta/info.json"
        info = json.loads(file.read_text(encoding="utf-8"))
        info["total_episodes"] = 500
        write_json(file, info)
        with self.assertRaisesRegex(ValueError, "exactly 50"):
            audit_data(self.root, self.manifest)

    def test_missing_camera_is_rejected(self):
        camera = self.root / "data/lerobot/adjust_bottle_joint_v30/videos" / CAMERAS[0] / "part.mp4"
        camera.unlink()
        with self.assertRaisesRegex(ValueError, "missing video"):
            audit_data(self.root, self.manifest)


class RecipeTests(unittest.TestCase):
    def test_ablation_changes_only_augmentation(self):
        p = cloud.paths(Path("/RoboTwin"), Path("/workspace/runtime/test"))
        base = {"data": {}, "model": {}, "train": {}}
        baseline = cloud.training_config(base, p, "ablation", "baseline", 8, 10000, 10000)
        robust = cloud.training_config(base, p, "ablation", "robust", 8, 10000, 10000)
        self.assertFalse(baseline["data"]["image_augment"])
        robust["data"]["image_augment"] = False
        self.assertEqual(robust, baseline)
        self.assertEqual(base, {"data": {}, "model": {}, "train": {}})
        self.assertFalse(baseline["train"]["use_lora"])
        self.assertTrue(baseline["train"]["align_params"]["video"]["use_patch_loss"])
        self.assertEqual(baseline["data"]["robot_config_root"], str(p["prepared"] / "robot_configs"))

    def test_four_and_eight_gpu_batch_invariant(self):
        for gpus in (4, 8):
            result = cloud.plan(gpus, 10000, 2500, 2)
            self.assertEqual(gpus * result["micro_batch_size"] * result["gradient_accumulation_steps"], 256)
            self.assertEqual(result["additional_free_disk_gib"], 300)
        with self.assertRaises(ValueError):
            cloud.plan(1, 10000, 2500, 2)

    def test_unsafe_run_name_rejected(self):
        with self.assertRaises(ValueError):
            cloud.training_config({"data": {}, "model": {}, "train": {}}, cloud.paths(Path("/RoboTwin"), Path("/tmp/r")),
                                  "../escape", "robust", 8, 10, 10)

    def test_resume_failure_raises_instead_of_restart(self):
        source = 'def run():\n    if True:\n        if True:\n' + ORIGINAL + '\n'
        scope = {"last_err": ValueError("broken checkpoint")}
        exec(strict_resume_source(source), scope)
        with self.assertRaisesRegex(RuntimeError, "refusing to restart"):
            scope["run"]()
        with self.assertRaises(ValueError):
            strict_resume_source("upstream changed")

    def test_preview_never_starts_gpu_work(self):
        with patch("robustwin.cloud.doctor", side_effect=AssertionError("GPU call")), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["train", "--run", "preview"]), 0)
        self.assertEqual(json.loads(out.getvalue())["mode"], "preview_only")

    def test_prepare_binds_robot_yaml_and_inference_to_same_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "RoboTwin"
            runtime = Path(directory) / "runtime"
            p = cloud.paths(root, runtime)
            mapping = p["source"] / "configs/robot_configs/robotwin.yaml"
            mapping.parent.mkdir(parents=True)
            mapping.write_text("norm_stats: mixed-data-stats.json\nstates: []\nactions: []\nimages: []\n", encoding="utf-8")
            audit = {"tasks": {task: {"path": f"/official/clean/{task}"} for task in TASKS}, "fingerprint": "fixture"}
            def fake_compute(*args, **kwargs):
                write_json(p["prepared"] / "norm_stats.json", norm_fixture())
            with patch("robustwin.cloud.audit_data", return_value=audit), patch("robustwin.cloud.run_logged", side_effect=fake_compute):
                cloud.prepare(root, runtime, "unused", 8)
            prepared_mapping = yaml.safe_load((p["prepared"] / "robot_configs/robotwin.yaml").read_text(encoding="utf-8"))
            config = cloud.training_config({"data": {}, "model": {}, "train": {}}, p, "test", "robust", 8, 10, 10)
            self.assertEqual(prepared_mapping["norm_stats"], config["data"]["norm_stats_file"])
            self.assertIn("mixed-data-stats.json", mapping.read_text(encoding="utf-8"))


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Install requirements-test.txt for Parquet integration checks")
class ParquetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / "data").mkdir()
        self.rows = [{"episode_index": episode, "frame_index": frame,
                      "observation.state": [0.0] * 14, "action": [0.0] * 14}
                     for episode in range(50) for frame in range(2)]

    def write(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(self.rows), self.path / "data/part.parquet")

    def test_all_rows_and_episode_counts_are_scanned(self):
        self.write()
        audit_rows(self.path, {"total_frames": 100})
        with self.assertRaisesRegex(ValueError, "totals"):
            audit_rows(self.path, {"total_frames": 99})

    def test_nan_action_is_rejected(self):
        self.rows[-1]["action"][4] = float("nan")
        self.write()
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            audit_rows(self.path, {"total_frames": 100})

    def test_duplicate_frame_is_rejected(self):
        self.rows[-1]["frame_index"] = 0
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_rows(self.path, {"total_frames": 100})

    def test_missing_episode_is_rejected(self):
        self.rows = self.rows[:-2]
        self.write()
        with self.assertRaisesRegex(ValueError, "totals"):
            audit_rows(self.path, {"total_frames": 98})


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "evaluation"
        result_fixture(self.run)

    def test_complete_separate_setting_scores(self):
        report = collect(self.run, 10)
        self.assertTrue(report["complete"])
        self.assertEqual(report["metrics"]["demo_clean"]["macro_success_rate"], 1)
        self.assertEqual(report["metrics"]["demo_randomized"]["macro_success_rate"], 0)
        self.assertEqual(report["clean_minus_randomized_gap"], 1)
        self.assertIsNone(report["official_score"])
        self.assertFalse(report["covers_100_episode_protocol"])

    def test_missing_task_never_inflates_mean(self):
        (self.run / "done/demo_clean__adjust_bottle.done").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            collect(self.run, 10)
        report = collect(self.run, 10, True)
        self.assertIsNone(report["metrics"]["demo_clean"]["macro_success_rate"])
        self.assertEqual(len(report["missing"]), 1)

    def test_repeated_final_line_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_result("Final success rate: 1/10 = 10%\nFinal success rate: 10/10 = 100%", 10)

    def test_counts_and_percentage_are_checked(self):
        for line in ("Final success rate: 11/10 = 110%", "Final success rate: 1/100 = 1%", "Final success rate: 1/10 = 90%"):
            with self.assertRaises(ValueError):
                parse_result(line, 10)
        self.assertEqual(parse_result("\x1b[32mFinal success rate: 1/3 = 33.33%\x1b[0m", 3), (1, 3))

    def test_unknown_task_rejected(self):
        (self.run / "done/demo_clean__fake_task.done").touch()
        with self.assertRaisesRegex(ValueError, "Unknown"):
            collect(self.run, 10)

    def test_report_writes_csv_json_and_html(self):
        report = write_report(self.run, 10)
        self.assertEqual(len(report["rows"]), 100)
        for name in ("report.json", "results.csv", "report.html"):
            self.assertTrue((self.run / name).is_file())
        self.assertIn("官方成绩待赛事组评定", (self.run / "report.html").read_text(encoding="utf-8"))

    def test_wilson_retains_uncertainty_at_extremes(self):
        low, high = wilson(0, 2)
        self.assertEqual(low, 0)
        self.assertGreater(high, 0.5)
        low, high = wilson(2, 2)
        self.assertLess(low, 0.5)
        self.assertAlmostEqual(high, 1)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.identity = model_fixture(self.model)

    def test_export_hash_detects_mutation(self):
        self.assertEqual(artifacts.verify_model(self.model), self.identity)
        (self.model / "config.json").write_text('{"changed":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed after export"):
            artifacts.verify_model(self.model)

    def test_weight_index_cannot_escape_model(self):
        write_json(self.model / "model.safetensors.index.json", {"weight_map": {"x": "../outside.safetensors"}})
        with self.assertRaisesRegex(ValueError, "escapes"):
            artifacts.weight_files(self.model)

    def test_missing_shard_is_rejected(self):
        write_json(self.model / "model.safetensors.index.json", {"weight_map": {"x": "missing.safetensors"}})
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            artifacts.weight_files(self.model)

    def test_bundle_refuses_another_models_scores(self):
        evaluation = self.root / "eval"
        result_fixture(evaluation, model_identity="another-model")
        with self.assertRaisesRegex(ValueError, "different model"):
            artifacts.bundle(self.model, evaluation, self.root / "candidate.zip")

    def test_bundle_contains_evidence_but_no_vendor_or_environment(self):
        evaluation = self.root / "eval"
        result_fixture(evaluation, model_identity=self.identity)
        output = self.root / "candidate.zip"
        result = artifacts.bundle(self.model, evaluation, output)
        self.assertIsNone(result["official_score"])
        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
            self.assertIn("submission-manifest.json", names)
            self.assertIn("model/model.safetensors", names)
            self.assertIn("evaluation/logs/eval_demo_clean__adjust_bottle.log", names)
            self.assertFalse(any("vendor/" in name or ".venv/" in name for name in names))
            manifest = json.loads(archive.read("submission-manifest.json"))
            self.assertEqual(manifest["self_eval_episodes"], 10)
            self.assertFalse(manifest["covers_100_episode_protocol"])


class PinnedSourceTests(unittest.TestCase):
    @unittest.skipUnless((PROJECT / "vendor").is_dir(), "Local reference checkout is not part of the portable source package")
    def test_contract_matches_downloaded_amd_sources(self):
        amd = PROJECT / "vendor" / ("Robotwin-radeon-cloud-" + LOCK["amd_reference"]["revision"])
        experiment = amd / "docker/assets/experiments/lingbot_vla_v2_6b_robotwin"
        for name, expected in LOCK["reference_files"].items():
            self.assertEqual(sha256(experiment / name), expected)
        lingbot = PROJECT / "vendor" / ("lingbot-vla-v2-" + LOCK["lingbot_revision"])
        trainer = lingbot / "tasks/vla/train_lingbotvla.py"
        self.assertEqual(sha256(trainer), LOCK["trainer_sha256"])
        for name, expected in LOCK["source_files"].items():
            self.assertEqual(sha256(lingbot / name), expected)
        compile(strict_resume_source(trainer.read_text(encoding="utf-8")), str(trainer), "exec")
        cloud.validate_norm(lingbot / "assets/norm_stats/robotwin.json")


if __name__ == "__main__":
    unittest.main()
