from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_builder import OLAPPromptBuilder


UNIQUE_NAME_PATTERN = re.compile(r"\[[^\]]+\](?:\.\[[^\]]+\])+")
METRIC_KEYS = [
    "total_time_seconds",
    "metadata_selection_total_seconds",
    "metadata_selection_dimensions_seconds",
    "metadata_selection_measures_seconds",
    "mdx_generation_seconds",
    "mdx_execution_seconds",
    "mdx_repair_generation_seconds",
    "mdx_repair_execution_seconds",
    "mdx_auto_repair_execution_seconds",
    "final_answer_generation_seconds",
    "vector_retrieval_seconds",
    "vector_retrieval_dimensions_seconds",
    "vector_retrieval_children_seconds",
    "vector_retrieval_measures_seconds",
]


@dataclass
class QuestionEval:
    question_id: str
    workflow: str
    question: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    task_success: int
    predicted: list[str]
    gold: list[str]
    timings: dict[str, float]
    actual_value: float | None = None
    gold_value: float | None = None
    error: str | None = None


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def compute_prf(predicted: set[str], gold: set[str]) -> tuple[int, int, int, float, float, float]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return tp, fp, fn, precision, recall, f1


def run_workflow(builder: OLAPPromptBuilder, workflow: str, question: str) -> dict[str, Any]:
    if workflow == "classic":
        return builder.run_classic(question)
    if workflow == "vector":
        return builder.run_vector(question)
    if workflow == "vector_hierarchical":
        return builder.run_vector_hierarchical(question)
    raise ValueError(f"Unsupported workflow: {workflow}")


def extract_unique_names(text: str) -> set[str]:
    if not text:
        return set()
    return {m.group(0).strip() for m in UNIQUE_NAME_PATTERN.finditer(text)}


def flatten_gold(item: dict[str, Any]) -> set[str]:
    if "gold_unique_names" in item:
        return {str(v).strip() for v in item["gold_unique_names"] if str(v).strip()}

    gold_by_type = item.get("gold_by_type", {})
    values: set[str] = set()
    if isinstance(gold_by_type, dict):
        for _, arr in gold_by_type.items():
            if isinstance(arr, list):
                for v in arr:
                    s = str(v).strip()
                    if s:
                        values.add(s)
    return values


def predicted_from_result(result: dict[str, Any]) -> set[str]:
    workflow = str(result.get("workflow", ""))
    if workflow == "classic":
        dims = extract_unique_names(str(result.get("dimensions_response", "")))
        meas = extract_unique_names(str(result.get("measures_response", "")))
        return dims | meas
    return extract_unique_names(str(result.get("metadata_response", "")))


def task_success_from_values(actual_value: float | None, gold_value: float | None) -> int:
    return 1 if actual_value is not None and gold_value is not None and actual_value == gold_value else 0


def build_failed_eval(
    question_id: str,
    workflow: str,
    question: str,
    gold: set[str],
    gold_value: Any,
    error: str,
) -> QuestionEval:
    return QuestionEval(
        question_id=question_id,
        workflow=workflow,
        question=question,
        tp=0,
        fp=0,
        fn=len(gold),
        precision=0.0,
        recall=0.0,
        f1=0.0,
        task_success=0,
        predicted=[],
        gold=sorted(gold),
        timings={},
        actual_value=None,
        gold_value=gold_value,
        error=error,
    )


def serialize_eval(eval_row: QuestionEval) -> dict[str, Any]:
    return {
        "question_id": eval_row.question_id,
        "workflow": eval_row.workflow,
        "question": eval_row.question,
        "tp": eval_row.tp,
        "fp": eval_row.fp,
        "fn": eval_row.fn,
        "precision": eval_row.precision,
        "recall": eval_row.recall,
        "f1": eval_row.f1,
        "task_success": eval_row.task_success,
        "predicted": eval_row.predicted,
        "gold": eval_row.gold,
        "actual_value": eval_row.actual_value,
        "gold_value": eval_row.gold_value,
        "values_match": eval_row.actual_value == eval_row.gold_value
        if eval_row.actual_value is not None and eval_row.gold_value is not None
        else None,
        "timings": eval_row.timings,
        "error": eval_row.error,
    }


def evaluate_question(builder: OLAPPromptBuilder, workflow: str, item: dict[str, Any], index: int) -> QuestionEval:
    qid = str(item.get("id", index))
    question = str(item["question"])
    gold = flatten_gold(item)
    gold_value = item.get("gold_value")

    try:
        result = run_workflow(builder, workflow, question)
    except Exception as exc:  # noqa: BLE001
        return build_failed_eval(qid, workflow, question, gold, gold_value, str(exc))

    predicted = predicted_from_result(result)
    tp, fp, fn, precision, recall, f1 = compute_prf(predicted, gold)
    actual_value_raw = result.get("singular_value")
    actual_value = float(actual_value_raw) if actual_value_raw is not None else None
    task_success = task_success_from_values(actual_value, gold_value)

    return QuestionEval(
        question_id=qid,
        workflow=workflow,
        question=question,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        task_success=task_success,
        predicted=sorted(predicted),
        gold=sorted(gold),
        timings={k: float(v) for k, v in result.get("timings", {}).items()},
        actual_value=actual_value,
        gold_value=gold_value,
        error=None,
    )


def aggregate(evals: list[QuestionEval], workflow: str) -> dict[str, Any]:
    rows = [e for e in evals if e.workflow == workflow]
    if not rows:
        return {"workflow": workflow, "questions": 0}

    def avg(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    totals = {
        key: float(sum(row.timings.get(key, 0.0) for row in rows))
        for key in METRIC_KEYS
    }
    avgs = {
        key: float(avg([row.timings.get(key, 0.0) for row in rows]))
        for key in METRIC_KEYS
    }

    return {
        "workflow": workflow,
        "questions": len(rows),
        "macro_precision": avg([r.precision for r in rows]),
        "macro_recall": avg([r.recall for r in rows]),
        "macro_f1": avg([r.f1 for r in rows]),
        "task_success_rate": avg([float(r.task_success) for r in rows]),
        "time_totals_seconds": totals,
        "time_averages_seconds": avgs,
    }


def run_analysis(questions_path: Path, workflows: list[str]) -> dict[str, Any]:
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Questions file must be a JSON list")

    builder = OLAPPromptBuilder()
    evals: list[QuestionEval] = []

    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item #{i} must be an object")
        if "question" not in item:
            raise ValueError(f"Item #{i} is missing question")

        for wf in workflows:
            print(f"[analysis] Running question #{i} with workflow={wf}")
            evals.append(evaluate_question(builder, wf, item, i))

    per_question = [serialize_eval(e) for e in evals]

    summaries = {wf: aggregate(evals, wf) for wf in workflows}

    return {
        "questions_file": str(questions_path),
        "workflows": workflows,
        "per_question": per_question,
        "summaries": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run thesis-style analysis over NL->OLAP workflows with precision/recall/F1, "
            "task success rate, and detailed timing totals/averages."
        )
    )
    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
        help=(
            "Path to JSON list of questions. Supported gold label formats: "
            "gold_unique_names or gold_by_type."
        ),
    )
    parser.add_argument(
        "--workflows",
        nargs="+",
        default=["classic", "vector", "vector_hierarchical"],
        help="Workflows to evaluate (classic, vector, vector_hierarchical)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chatbot/analysis_results.json"),
        help="Output JSON file path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(args.questions, args.workflows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Macro Summary ===")
    for wf in args.workflows:
        s = result["summaries"].get(wf, {})
        if s.get("questions", 0) == 0:
            print(f"{wf}: no data")
            continue
        print(
            f"{wf}: questions={s['questions']} "
            f"macro_precision={s['macro_precision']:.4f} "
            f"macro_recall={s['macro_recall']:.4f} "
            f"macro_f1={s['macro_f1']:.4f} "
            f"task_success_rate={s['task_success_rate']:.4f} "
            f"avg_total_time_s={s['time_averages_seconds'].get('total_time_seconds', 0.0):.4f}"
        )

    print(f"\nSaved analysis to: {args.output}")


if __name__ == "__main__":
    main()
