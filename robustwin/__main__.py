from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import artifacts, cloud
from .common import now, write_json
from .data import audit_data
from .results import write_report


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RobustTwin · clean-only LingBot-VLA 2.0 competition pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan", help="Estimate credits and disk without starting GPU work")
    p.add_argument("--gpus", type=int, choices=(4, 8), default=8)
    p.add_argument("--steps", type=positive, default=10000)
    p.add_argument("--save-every", type=positive, default=10000)
    p.add_argument("--episodes", type=positive, default=2)
    for name in ("doctor", "prepare", "train", "export", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--root", type=Path, default=Path("/RoboTwin"))
        sub.add_argument("--runtime", type=Path, default=Path("/workspace/runtime/robustwin"))
        sub.add_argument("--python", default="/opt/robotwin-env/bin/python")
        sub.add_argument("--gpus", type=int, choices=(4, 8), default=8)
        if name != "doctor":
            sub.add_argument("--execute", action="store_true", help="Execute in the AMD image; otherwise preview only")
        if name in ("train", "export", "evaluate"):
            sub.add_argument("--run", required=True)
        if name == "train":
            sub.add_argument("--recipe", choices=("baseline", "robust"), default="robust")
            sub.add_argument("--steps", type=positive, default=10000)
            sub.add_argument("--save-every", type=positive, default=10000)
            sub.add_argument("--resume-step", type=positive)
        if name == "evaluate":
            sub.add_argument("--model", type=Path, required=True)
            sub.add_argument("--episodes", type=positive, default=2)
            sub.add_argument("--resume", action="store_true")
    p = commands.add_parser("audit", help="Audit AMD clean LeRobot datasets; --deep scans all action/state rows")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--deep", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("report", help="Summarize real benchmark logs")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--episodes", type=positive, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    p = commands.add_parser("bundle", help="Package trained model, complete self-evaluation and source")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = cloud.plan(args.gpus, args.steps, args.save_every, args.episodes)
        elif args.command == "audit":
            result = audit_data(args.root, args.manifest, deep=args.deep)
            write_json(args.out, result)
        elif args.command == "report":
            result = write_report(args.run_dir, args.episodes, args.allow_incomplete)
            result = {key: value for key, value in result.items() if key != "rows"}
        elif args.command == "bundle":
            result = artifacts.bundle(args.model.resolve(), args.evaluation.resolve(), args.out.resolve())
        elif args.command != "doctor" and not args.execute:
            result = {"mode": "preview_only", "action": args.command,
                      "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      "next": "Use --execute inside the pinned AMD robotwin image after doctor succeeds."}
            if args.command == "train":
                result["resources"] = cloud.plan(args.gpus, args.steps, args.save_every, 2)
        else:
            root, runtime = args.root.resolve(), args.runtime.resolve()
            required_disk = 4
            if args.command == "train":
                remaining = args.steps - (args.resume_step or 0)
                required_disk = cloud.plan(args.gpus, remaining, args.save_every, 2)["additional_free_disk_gib"]
            elif args.command == "export":
                required_disk = 20
            environment = cloud.doctor(root, runtime, args.python, args.gpus, required_disk)
            if args.command == "doctor":
                result = environment
            else:
                write_json(runtime / "environment" / (now().replace(":", "-") + ".json"), environment)
                if args.command == "prepare":
                    result = cloud.prepare(root, runtime, args.python, args.gpus)
                elif args.command == "train":
                    result = cloud.train(root, runtime, args.python, args.gpus, args.run, args.recipe,
                                         args.steps, args.save_every, args.resume_step, environment)
                elif args.command == "export":
                    result = artifacts.export_model(root, runtime, args.python, args.run)
                else:
                    result = artifacts.evaluate(root, runtime, args.python, args.gpus, args.run,
                                                args.model, args.episodes, args.resume)
                    result = {key: value for key, value in result.items() if key != "rows"}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f"RobustTwin stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
