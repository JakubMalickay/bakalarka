from __future__ import annotations

import argparse
import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_builder import OLAPPromptBuilder
from chatbot_initiation import load_chatbot_config


WORKFLOWS = ("classic", "vector", "vector_hierarchical", "vector_two_tier")
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
class UserSession:
    user_id: str
    questions: list[str]
    workflow: str
    embedding_model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-user sessions for all three workflows with aggregate metrics."
    )
    parser.add_argument("--questions", type=Path, required=True, help="Questions JSON file.")
    parser.add_argument("--users", type=Path, required=True, help="Users JSON file (list of strings).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chatbot/multi_user_results.json"),
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent user sessions.",
    )
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=None,
        help="Embedding models to run. If omitted, uses config.yaml metadata_selection.embedding_models.",
    )
    return parser.parse_args()


def resolve_embedding_models(cli_models: list[str] | None) -> list[str]:
    if cli_models:
        return [model.strip() for model in cli_models if model.strip()]

    config = load_chatbot_config()
    selection = config.get("metadata_selection", {})

    models = selection.get("embedding_models")
    if isinstance(models, list):
        cleaned = [str(model).strip() for model in models if str(model).strip()]
        if cleaned:
            return cleaned

    explicit_model = str(selection.get("embedding_model", "")).strip()
    if explicit_model:
        return [explicit_model]

    preset = str(selection.get("embedding_model_preset", "default")).strip().lower()
    if preset == "strong":
        return ["BAAI/bge-m3"]
    return ["BAAI/bge-large-en-v1.5"]


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


def extract_unique_names(text: str) -> set[str]:
    return {m.group(0).strip() for m in UNIQUE_NAME_PATTERN.finditer(text or "")}


def flatten_gold(item: dict[str, Any]) -> set[str]:
    return {str(v).strip() for v in item["gold_unique_names"] if str(v).strip()}


def task_success_from_values(actual_value: float | None, gold_value: float | None) -> int:
    return 1 if actual_value is not None and gold_value is not None and actual_value == gold_value else 0


def load_users(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [str(user).strip() for user in raw]


def load_question_items(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_question_index(questions_path: Path) -> dict[str, dict[str, Any]]:
    items = load_question_items(questions_path)
    return {str(item["question"]).strip(): item for item in items}


def build_user_sessions(
    users_path: Path,
    question_index: dict[str, dict[str, Any]],
    embedding_models: list[str],
) -> list[UserSession]:
    users = load_users(users_path)
    questions = list(question_index.keys())

    sessions: list[UserSession] = []
    for embedding_model in embedding_models:
        for user_id in users:
            for workflow in WORKFLOWS:
                sessions.append(
                    UserSession(
                        user_id=user_id,
                        questions=list(questions),
                        workflow=workflow,
                        embedding_model=embedding_model,
                    )
                )
    return sessions


def run_workflow(builder: OLAPPromptBuilder, workflow: str, question: str) -> dict[str, Any]:
    if workflow == "classic":
        return builder.run_classic(question)
    if workflow == "vector":
        return builder.run_vector(question)
    if workflow == "vector_hierarchical":
        return builder.run_vector_hierarchical(question)
    if workflow == "vector_two_tier":
        return builder.run_vector_two_tier(question)
    raise ValueError(f"Unsupported workflow: {workflow}")


def extract_predicted(result: dict[str, Any], workflow: str) -> set[str]:
    if workflow == "classic":
        dims = extract_unique_names(str(result.get("dimensions_response", "")))
        meas = extract_unique_names(str(result.get("measures_response", "")))
        return dims | meas
    return extract_unique_names(str(result.get("metadata_response", "")))


def run_user_session(session: UserSession, question_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    builder = OLAPPromptBuilder(selection_overrides={"embedding_model": session.embedding_model})
    question_results: list[dict[str, Any]] = []

    for index, question in enumerate(session.questions, start=1):
        question_data = question_index[question]
        gold = flatten_gold(question_data)
        gold_value = question_data.get("gold_value")

        try:
            result = run_workflow(builder, session.workflow, question)
            predicted = extract_predicted(result, session.workflow)
            tp, fp, fn, precision, recall, f1 = compute_prf(predicted, gold)

            actual_value_raw = result.get("singular_value")
            actual_value = float(actual_value_raw) if actual_value_raw is not None else None
            task_success = task_success_from_values(actual_value, gold_value)

            question_results.append(
                {
                    "question_index": index,
                    "question_id": question_data.get("id", f"q{index:02d}"),
                    "question": question,
                    "singular_value": result.get("singular_value"),
                    "actual_value": actual_value,
                    "gold_value": gold_value,
                    "values_match": actual_value == gold_value
                    if actual_value is not None and gold_value is not None
                    else None,
                    "final_answer": result.get("final_answer"),
                    "executed_mdx_query": result.get("executed_mdx_query"),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "task_success": task_success,
                    "predicted": sorted(predicted),
                    "gold": sorted(gold),
                    "timings": result.get("timings", {}),
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            question_results.append(
                {
                    "question_index": index,
                    "question_id": question_data.get("id", f"q{index:02d}"),
                    "question": question,
                    "singular_value": None,
                    "actual_value": None,
                    "gold_value": gold_value,
                    "values_match": None,
                    "final_answer": None,
                    "executed_mdx_query": None,
                    "tp": 0,
                    "fp": 0,
                    "fn": len(gold),
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "task_success": 0,
                    "predicted": [],
                    "gold": sorted(gold),
                    "timings": {},
                    "error": str(exc),
                }
            )

    return {
        "user_id": session.user_id,
        "workflow": session.workflow,
        "embedding_model": session.embedding_model,
        "questions": question_results,
        "session_metrics": aggregate_metrics(question_results),
    }


def aggregate_metrics(question_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not question_results:
        return {}

    def avg(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    total_time = {}
    avg_time = {}
    for key in METRIC_KEYS:
        times = [q.get("timings", {}).get(key, 0.0) for q in question_results]
        total_time[key] = sum(times)
        avg_time[key] = avg(times)

    return {
        "num_questions": len(question_results),
        "total_tp": sum(q.get("tp", 0) for q in question_results),
        "total_fp": sum(q.get("fp", 0) for q in question_results),
        "total_fn": sum(q.get("fn", 0) for q in question_results),
        "macro_precision": avg([q.get("precision", 0.0) for q in question_results]),
        "macro_recall": avg([q.get("recall", 0.0) for q in question_results]),
        "macro_f1": avg([q.get("f1", 0.0) for q in question_results]),
        "task_success_rate": avg([float(q.get("task_success", 0)) for q in question_results]),
        "time_totals_seconds": total_time,
        "time_averages_seconds": avg_time,
    }


def run_sessions(
    sessions: list[UserSession], max_workers: int, question_index: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    # Run models sequentially to keep vector DB rebuilds deterministic per model.
    sessions_by_model: dict[str, list[UserSession]] = {}
    for session in sessions:
        sessions_by_model.setdefault(session.embedding_model, []).append(session)

    for embedding_model, model_sessions in sessions_by_model.items():
        print(f"[multi-user] Running model: {embedding_model}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_user_session, session, question_index) for session in model_sessions]
            for future in as_completed(futures):
                results.append(future.result())

    return results


def aggregate_overall(sessions_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_questions: list[dict[str, Any]] = []
    for session in sessions_results:
        all_questions.extend(session.get("questions", []))
    return aggregate_metrics(all_questions)


def aggregate_by_workflow(sessions_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {wf: [] for wf in WORKFLOWS}
    for session in sessions_results:
        grouped[session["workflow"]].extend(session.get("questions", []))
    return {workflow: aggregate_metrics(rows) for workflow, rows in grouped.items()}


def aggregate_by_model(sessions_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for session in sessions_results:
        grouped.setdefault(session["embedding_model"], []).extend(session.get("questions", []))
    return {model: aggregate_metrics(rows) for model, rows in grouped.items()}


def aggregate_by_workflow_model(sessions_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for session in sessions_results:
        key = f"{session['embedding_model']}::{session['workflow']}"
        grouped.setdefault(key, []).extend(session.get("questions", []))
    return {key: aggregate_metrics(rows) for key, rows in grouped.items()}


def aggregate_by_user(sessions_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for session in sessions_results:
        grouped.setdefault(session["user_id"], []).extend(session.get("questions", []))
    return {user: aggregate_metrics(rows) for user, rows in grouped.items()}


def main() -> None:
    args = parse_args()

    embedding_models = resolve_embedding_models(args.embedding_models)
    question_index = build_question_index(args.questions)
    sessions = build_user_sessions(args.users, question_index, embedding_models)
    session_results = run_sessions(sessions, args.max_workers, question_index)

    results = {
        "embedding_models": embedding_models,
        "sessions": session_results,
        "overall_metrics": aggregate_overall(session_results),
        "metrics_by_workflow": aggregate_by_workflow(session_results),
        "metrics_by_model": aggregate_by_model(session_results),
        "metrics_by_workflow_model": aggregate_by_workflow_model(session_results),
        "metrics_by_user": aggregate_by_user(session_results),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = results["overall_metrics"]
    print("\n=== Overall Metrics ===")
    print(f"Num questions: {metrics.get('num_questions', 'N/A')}")
    print(f"Macro Precision: {metrics.get('macro_precision', 0.0):.3f}")
    print(f"Macro Recall: {metrics.get('macro_recall', 0.0):.3f}")
    print(f"Macro F1: {metrics.get('macro_f1', 0.0):.3f}")
    print(f"Task Success Rate: {metrics.get('task_success_rate', 0.0):.1%}")
    print("\n=== By Model ===")
    for model in embedding_models:
        m_metrics = results["metrics_by_model"].get(model, {})
        print(f"{model}:")
        print(f"  Precision: {m_metrics.get('macro_precision', 0.0):.3f}")
        print(f"  Recall: {m_metrics.get('macro_recall', 0.0):.3f}")
        print(f"  F1: {m_metrics.get('macro_f1', 0.0):.3f}")
        print(f"  Success Rate: {m_metrics.get('task_success_rate', 0.0):.1%}")

    print("\n=== By Workflow ===")
    for workflow in WORKFLOWS:
        wf_metrics = results["metrics_by_workflow"].get(workflow, {})
        print(f"{workflow}:")
        print(f"  Precision: {wf_metrics.get('macro_precision', 0.0):.3f}")
        print(f"  Recall: {wf_metrics.get('macro_recall', 0.0):.3f}")
        print(f"  F1: {wf_metrics.get('macro_f1', 0.0):.3f}")
        print(f"  Success Rate: {wf_metrics.get('task_success_rate', 0.0):.1%}")

    print("\n=== By Workflow + Model ===")
    for model in embedding_models:
        for workflow in WORKFLOWS:
            key = f"{model}::{workflow}"
            wm_metrics = results["metrics_by_workflow_model"].get(key, {})
            print(f"{model} | {workflow}:")
            print(f"  Precision: {wm_metrics.get('macro_precision', 0.0):.3f}")
            print(f"  Recall: {wm_metrics.get('macro_recall', 0.0):.3f}")
            print(f"  F1: {wm_metrics.get('macro_f1', 0.0):.3f}")
            print(f"  Success Rate: {wm_metrics.get('task_success_rate', 0.0):.1%}")

    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
