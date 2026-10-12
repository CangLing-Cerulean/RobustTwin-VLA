from __future__ import annotations

import csv
import html
import math
import re
from pathlib import Path

from .common import SETTINGS, TASKS, now, read_json, write_json

FINAL = re.compile(r"Final success rate:\s*(\d+)/(\d+)\s*=\s*([0-9.]+)%")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_result(text: str, expected_episodes: int) -> tuple[int, int]:
    matches = FINAL.findall(ANSI.sub("", text))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one final result, found {len(matches)}.")
    successes, total, percent = int(matches[0][0]), int(matches[0][1]), float(matches[0][2])
    if total != expected_episodes or not 0 <= successes <= total:
        raise ValueError(f"Invalid result {successes}/{total}; expected {expected_episodes} episodes.")
    if not math.isfinite(percent) or abs(percent - successes / total * 100) > 0.011:
        raise ValueError("Printed success percentage does not match counts.")
    return successes, total


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval needs valid binomial counts.")
    z = 1.959963984540054
    p = successes / total
    divisor = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / divisor
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / divisor
    return max(0.0, centre - margin), min(1.0, centre + margin)


def collect(run: Path, episodes: int, allow_incomplete: bool = False) -> dict:
    if episodes <= 0:
        raise ValueError("Episode count must be positive.")
    expected = {f"{setting}__{task}" for setting in SETTINGS for task in TASKS}
    actual_logs = {path.stem.removeprefix("eval_") for path in (run / "logs").glob("eval_*.log")}
    actual_markers = {path.stem for path in (run / "done").glob("*.done")}
    unknown = (actual_logs | actual_markers) - expected
    if unknown:
        raise ValueError(f"Unknown evaluation tasks: {sorted(unknown)}")
    rows, missing, invalid = [], [], []
    for setting in SETTINGS:
        for task in TASKS:
            key = f"{setting}__{task}"
            log = run / "logs" / f"eval_{key}.log"
            if key not in actual_markers or not log.is_file():
                missing.append(key)
                continue
            try:
                successes, total = parse_result(log.read_text(encoding="utf-8", errors="strict"), episodes)
            except (ValueError, UnicodeError) as exc:
                invalid.append({"task": key, "error": str(exc)})
                continue
            low, high = wilson(successes, total)
            rows.append({"task": task, "setting": setting, "successes": successes, "episodes": total,
                         "success_rate": successes / total, "wilson_95_low": low, "wilson_95_high": high})
    complete = len(rows) == len(expected) and not missing and not invalid
    if not complete and not allow_incomplete:
        raise ValueError(f"Incomplete evaluation: {len(rows)}/100 valid results; {len(missing)} missing; {len(invalid)} invalid. Use --allow-incomplete only for inspection.")
    metrics = {}
    for setting in SETTINGS:
        selected = [row for row in rows if row["setting"] == setting]
        metrics[setting] = {"completed_tasks": len(selected),
                            "macro_success_rate": sum(row["success_rate"] for row in selected) / 50 if len(selected) == 50 else None,
                            "successes": sum(row["successes"] for row in selected),
                            "episodes": sum(row["episodes"] for row in selected)}
    clean, random = (metrics[s]["macro_success_rate"] for s in SETTINGS)
    evidence = read_json(run / "robustwin-eval.json") if (run / "robustwin-eval.json").is_file() else None
    if evidence and evidence["episodes"] != episodes:
        raise ValueError("Report episode count differs from the recorded evaluation configuration.")
    return {"schema_version": 1, "generated_at": now(), "type": "self_evaluation", "official_score": None,
            "complete": complete, "valid_task_settings": len(rows), "expected_task_settings": 100,
            "episodes_per_task_setting": episodes, "covers_100_episode_protocol": complete and episodes == 100,
            "metrics": metrics, "clean_minus_randomized_gap": clean - random if clean is not None and random is not None else None,
            "rows": rows, "missing": missing, "invalid": invalid, "run": evidence,
            "notes": "Self-evaluation only. Wilson intervals describe individual task binomial sampling, not aggregate uncertainty across tasks. No official ranking formula is assumed."}


def write_report(run: Path, episodes: int, allow_incomplete: bool = False) -> dict:
    report = collect(run, episodes, allow_incomplete)
    write_json(run / "report.json", report)
    with (run / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["task", "setting", "successes", "episodes", "success_rate", "wilson_95_low", "wilson_95_high"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["rows"])
    by_task = {(row["task"], row["setting"]): row for row in report["rows"]}
    def cell(task, setting):
        row = by_task.get((task, setting))
        if row is None:
            return '<td class="missing">待完成</td>'
        percent = row["success_rate"] * 100
        return (f'<td><b>{percent:.1f}%</b><span>{row["successes"]} / {row["episodes"]}</span>'
                f'<div class="bar"><i style="width:{percent:.2f}%"></i></div></td>')
    table = "".join(f'<tr><th>{html.escape(task)}</th>{cell(task, SETTINGS[0])}{cell(task, SETTINGS[1])}</tr>' for task in TASKS)
    cards = ""
    for setting, title in zip(SETTINGS, ("Clean", "Randomized")):
        metric = report["metrics"][setting]
        value = "待完成" if metric["macro_success_rate"] is None else f'{100 * metric["macro_success_rate"]:.2f}%'
        cards += f'<article><label>{title}</label><strong>{value}</strong><p>{metric["completed_tasks"]} / 50 任务</p></article>'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RobustTwin · 评测记录</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0b1014;color:#e3e7e9;font:15px/1.6 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:56px 24px}}header{{border-top:1px solid #46505a;padding-top:20px}}.eyebrow{{font:12px monospace;letter-spacing:.18em;color:#b4ed71}}
h1{{font-size:clamp(38px,6vw,74px);letter-spacing:-.055em;line-height:1.1;margin:28px 0}}p{{color:#99a6ae}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:40px 0}}
article{{padding:28px;border:1px solid #313c43;border-radius:4px}}label{{color:#a9b4bb}}strong{{display:block;font-size:44px;font-weight:500}}article p{{margin:0}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:18px 12px;border-bottom:1px solid #253138}}th{{font-weight:450}}thead{{color:#a9b4bb}}td{{width:25%}}td b{{font-weight:500}}td span{{float:right;color:#87959e;font:12px monospace}}
.bar{{height:2px;background:#27343b;margin-top:8px}}.bar i{{display:block;height:2px;background:#b4ed71}}.missing{{color:#bf9c6e}}footer{{margin-top:38px;color:#82909a;font-size:12px}}
@media(max-width:600px){{main{{padding:28px 12px}}th,td{{padding:14px 5px;font-size:12px}}td span{{display:block;float:none}}strong{{font-size:32px}}}}
</style><main><header><div class="eyebrow">ROBUSTTWIN / EVALUATION RECORD</div><h1>从 Clean，走向泛化。</h1>
<p>本地自测 · {report["valid_task_settings"]} / 100 项完成 · 每项 {episodes} 次 · 官方成绩待赛事组评定</p></header>
<section class="cards">{cards}</section><table><thead><tr><th>任务</th><th>Clean</th><th>Randomized</th></tr></thead><tbody>{table}</tbody></table>
<footer>仅完整的 50 项结果计算各 setting 的宏平均。缺失或无效结果保留为待完成。<br>{html.escape(report["generated_at"])}</footer></main></html>'''
    (run / "report.html").write_text(page, encoding="utf-8")
    return report
