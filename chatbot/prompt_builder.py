"""Two-step prompt builder for OLAP-aware chatbot.

Step 1 – Dimensions pass:
    Combines the user question with the system prompt and all available
    dimensions / hierarchies / levels / attributes from cube_dimensions.json.
    The LLM is asked to identify which dimensions are relevant.

Step 2 – Measures pass:
    Combines the user question with the system prompt and all available
    measures from cube_measures.json.
    The LLM is asked to identify which measures are relevant.

Usage:
    from chatbot.prompt_builder import OLAPPromptBuilder

    builder = OLAPPromptBuilder()
    result = builder.run("Show me internet sales by country for 2023")
    print(result["dimensions_response"])
    print(result["measures_response"])
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import math
import re
import time

# Allow running as a top-level script from any directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from chatbot.chatbot_initiation import build_client, load_chatbot_config  # noqa: E402
from OLAP.OLAP_connection import build_default_awdw, build_from_env  # noqa: E402

# ── JSON paths ────────────────────────────────────────────────────────────────
_OLAP_DIR = _REPO_ROOT / "OLAP"
_DIMENSIONS_JSON = _OLAP_DIR / "cube_dimensions.json"
_MEASURES_JSON = _OLAP_DIR / "cube_measures.json"

# ── System prompts ────────────────────────────────────────────────────────────
_SYSTEM_DIMENSIONS = (
    "You are an OLAP query assistant for the Adventure Works cube. "
    "Your goal is to help build an MDX query that answers the user's question. "
    "You must also create the final MDX query. "
    "Your task in this step is to identify which dimensions, hierarchies, levels, "
    "and attributes from the provided catalogue are needed in the MDX query. "
    "If the question can be answered directly by a measure without slicing/filtering/grouping/ranking, "
    "you may omit all dimensions, levels, and attributes. "
    "Return only the chosen objects grouped by type. "
    "Use exactly these headings when needed: DIMENSIONS, LEVELS, ATTRIBUTES. "
    "Under each heading list only the unique_name values exactly as given, one per line. "
    "Do not include explanations, numbering, commentary, code fences, or any extra text."
)

_SYSTEM_MEASURES = (
    "You are an OLAP query assistant for the Adventure Works cube. "
    "Your goal is to help build an MDX query that answers the user's question. "
    "You must also create the final MDX query. "
    "Your task in this step is to identify which measures from the provided catalogue "
    "are needed in the MDX query. "
    "For scalar total questions, prefer using an existing base measure directly instead of adding extra context. "
    "Return only the chosen objects grouped by type. "
    "Use exactly this heading: MEASURES. "
    "Under the heading list only the unique_name values exactly as given, one per line. "
    "Do not include explanations, numbering, commentary, code fences, or any extra text."
)

_SYSTEM_VECTOR_METADATA = (
    "You are an OLAP query assistant for the Adventure Works cube. "
    "Your goal is to help build an MDX query that answers the user's question. "
    "You must also create the final MDX query. "
    "You are given the top metadata candidates retrieved from a vector search over all OLAP objects. "
    "Review the full JSON objects and choose only the objects needed for the MDX query. "
    "Do not select dimensions/levels/attributes unless they are required by the user intent "
    "(for example filters, grouping, top-N, ranking, or breakdown). "
    "If a single existing measure answers the question, return only MEASURES. "
    "Return only the chosen objects grouped by type. "
    "Use exactly these headings when needed: DIMENSIONS, LEVELS, ATTRIBUTES, MEASURES. "
    "Under each heading list only the unique_name values exactly as given, one per line. "
    "Do not include explanations, numbering, commentary, code fences, or any extra text."
)

_SYSTEM_MDX_QUERY = (
    "You are an OLAP query assistant for the Adventure Works cube. "
    "Build a valid MDX query that answers the user's question using the selected objects. "
    "Always use FROM [Adventure Works] as the cube name. "
    "Do not create calculated members that reuse names of existing measures. "
    "If the user asks for a scalar total and no slicing/grouping is needed, prefer a direct query like "
    "SELECT {[Measures].[...]} ON COLUMNS FROM [Adventure Works]. "
    "Prefer selecting existing measures directly on COLUMNS. "
    "Only use WITH MEMBER when absolutely necessary and always give it a unique name not present in the cube. "
    "Output only the MDX query text. Do not include explanations, markdown, or extra text."
)

_SYSTEM_FINAL_ANSWER = (
    "You are an OLAP answer formatter. "
    "You are given a user question and a single numeric result from an OLAP query. "
    "Return one concise sentence that directly answers the question using that value. "
    "Do not add extra assumptions, caveats, or additional numbers."
)

_SYSTEM_MDX_SCALAR_REPAIR = (
    "You are an OLAP MDX repair assistant for Adventure Works cube. "
    "Given a user question and a previous MDX query, produce a corrected MDX query that returns a single numeric scalar value. "
    "Use one measure on COLUMNS and avoid returning row axes unless strictly required. "
    "Use hierarchy expressions with .MEMBERS or ALLMEMBERS only when the argument is truly a hierarchy, not a member. "
    "Do not create calculated members with names that collide with existing measures. "
    "Output only MDX text."
)


# ── Catalogue formatters ───────────────────────────────────────────────────────

def _format_dimensions(data: list[dict[str, Any]]) -> str:
    """Convert cube_dimensions.json into a compact, readable catalogue string."""
    cube = data[0]
    lines: list[str] = [f"Cube: {cube['cube']}\n"]

    for dim in cube.get("dimensions", []):
        lines.append(f"DIMENSION: {dim['name']}  ({dim['unique_name']})")
        lines.append(f"  Description: {dim.get('description', '')}")

        levels = dim.get("levels", [])
        if levels:
            lines.append("  Levels:")
            for lvl in levels:
                lines.append(
                    f"    - {lvl['name']}  unique_name={lvl['unique_name']}"
                    f"  | {lvl.get('description', '')}"
                )

        attributes = dim.get("attributes", [])
        if attributes:
            lines.append("  Attributes:")
            for attr in attributes:
                lines.append(
                    f"    - {attr['name']}  unique_name={attr['unique_name']}"
                    f"  | {attr.get('description', '')}"
                )
        lines.append("")

    return "\n".join(lines)


def _format_measures(data: list[dict[str, Any]]) -> str:
    """Convert cube_measures.json into a compact, readable catalogue string."""
    cube = data[0]
    lines: list[str] = [f"Cube: {cube['cube']}\n", "Measures:"]

    current_group: str | None = None
    for m in cube.get("measures", []):
        group = m.get("measure_group", "")
        if group != current_group:
            lines.append(f"\n  [{group}]")
            current_group = group
        lines.append(
            f"    - {m['name']}  unique_name={m['unique_name']}"
            f"  | {m.get('description', '')}"
        )

    return "\n".join(lines)


# ── Main builder class ────────────────────────────────────────────────────────

class OLAPPromptBuilder:
    """Runs a two-step LLM pipeline to extract relevant OLAP dimensions and measures."""

    _EMBEDDING_MODEL_PRESETS = {
        "default": "BAAI/bge-large-en-v1.5",
        "strong": "BAAI/bge-m3",
    }

    def __init__(
        self,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        selection_overrides: dict[str, Any] | None = None,
    ) -> None:
        self._config = load_chatbot_config()
        self._selection_config = dict(self._config.get("metadata_selection", {}))
        if selection_overrides:
            self._selection_config.update(selection_overrides)
        self._workflow = str(self._selection_config.get("workflow", "classic")).strip().lower()
        self._top_k = int(self._selection_config.get("top_k", 20))
        self._dimension_top_k = int(self._selection_config.get("dimension_top_k", 5))
        self._child_top_k_per_dimension = int(self._selection_config.get("child_top_k_per_dimension", 5))
        self._measure_top_k = int(self._selection_config.get("measure_top_k", 10))
        self._prefer_scalar_template = bool(self._selection_config.get("prefer_scalar_template", True))
        self._dims_catalogue = self._load_catalogue(_DIMENSIONS_JSON, "dimensions")
        self._measures_catalogue = self._load_catalogue(_MEASURES_JSON, "measures")
        self._vector_store = None
        self._client = build_client(temperature=temperature, max_tokens=max_tokens)

        try:
            self._olap_conn = build_from_env()
            print("[olap] Connected using environment configuration")
        except Exception:
            self._olap_conn = build_default_awdw()
            print("[olap] Connected using default Adventure Works configuration")

    # ── loaders ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_catalogue(path: Path, label: str) -> str:
        if not path.exists():
            raise FileNotFoundError(f"OLAP catalogue not found: {path}")
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        text = _format_dimensions(data) if label == "dimensions" else _format_measures(data)
        out_path = path.with_suffix(".txt")
        out_path.write_text(text, encoding="utf-8")
        print(f"[catalogue] Saved compact {label} to {out_path}")
        return text

    def _get_vector_store(self):
        if self._vector_store is None:
            from chatbot.metadata_vector_search import OLAPMetadataVectorStore  # noqa: E402

            embedding_model = self._resolve_embedding_model()

            self._vector_store = OLAPMetadataVectorStore(
                dimensions_path=_DIMENSIONS_JSON,
                measures_path=_MEASURES_JSON,
                db_dir=_REPO_ROOT / str(self._selection_config.get("vector_db_dir", "chatbot/vector_db")),
                embedding_model=embedding_model,
                embedding_device=str(self._selection_config.get("embedding_device", "auto")).strip().lower(),
                vector_dimensions=int(self._selection_config.get("vector_dimensions", 512)),
            )
        return self._vector_store

    def _resolve_embedding_model(self) -> str:
        explicit_model = str(self._selection_config.get("embedding_model", "")).strip()
        if explicit_model:
            return explicit_model

        preset = str(self._selection_config.get("embedding_model_preset", "default")).strip().lower()
        return self._EMBEDDING_MODEL_PRESETS.get(preset, self._EMBEDDING_MODEL_PRESETS["default"])

    @staticmethod
    def _serialize_vector_results(results: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "rank": result.rank,
                "score": result.score,
                "object_type": result.object_type,
                "name": result.name,
                "unique_name": result.unique_name,
            }
            for result in results
        ]

    # ── prompt constructors ───────────────────────────────────────────────────

    def _build_dimensions_prompt(self, user_question: str) -> str:
        return (
            "Below is the full catalogue of dimensions, hierarchies, levels, and attributes "
            "available in the Adventure Works OLAP cube.\n\n"
            f"{self._dims_catalogue}\n"
            "---\n"
            f"User question: {user_question}\n\n"
            "Select only the relevant dimensions, levels, and attributes for the MDX query, "
            "and create the MDX query as well. "
            "Return only headings plus unique_name lines, with no explanation."
        )

    def _build_measures_prompt(self, user_question: str) -> str:
        return (
            "Below is the full catalogue of measures available in the Adventure Works OLAP cube.\n\n"
            f"{self._measures_catalogue}\n"
            "---\n"
            f"User question: {user_question}\n\n"
            "Select only the relevant measures for the MDX query, and create the MDX query as well. "
            "Return only the MEASURES heading plus unique_name lines, with no explanation."
        )

    @staticmethod
    def _build_vector_prompt(user_question: str, candidates: list[dict[str, Any]]) -> str:
        candidate_json = json.dumps(candidates, indent=2, ensure_ascii=False)
        return (
            "Below are the top OLAP metadata candidates returned by a vector search over all "
            "dimensions, levels, attributes, and measures in the Adventure Works cube.\n\n"
            f"User question: {user_question}\n\n"
            "Top candidate JSON objects:\n"
            f"{candidate_json}\n\n"
            "Choose only the objects needed to build the MDX query for this question and create the MDX query. "
            "If the question is answerable by an existing measure alone, do not select any dimensions, levels, "
            "or attributes. "
            "Return only headings plus unique_name lines grouped by type, with no explanation."
        )

    @staticmethod
    def _build_mdx_prompt(user_question: str, selected_objects_text: str) -> str:
        return (
            f"User question: {user_question}\n\n"
            "Selected objects (grouped by type):\n"
            f"{selected_objects_text}\n\n"
            "Create a single valid MDX query that answers the question using these selected objects. "
            "Output only the MDX query."
        )

    @staticmethod
    def _build_final_answer_prompt(user_question: str, singular_value: float) -> str:
        return (
            f"User question: {user_question}\n"
            f"Single OLAP result value: {singular_value}\n\n"
            "Write one concise sentence that directly answers the user question using this value."
        )

    @staticmethod
    def _build_scalar_repair_prompt(user_question: str, failed_mdx: str) -> str:
        return (
            f"User question: {user_question}\n\n"
            "Previous MDX query that did not yield a usable scalar numeric value:\n"
            f"{failed_mdx}\n\n"
            "Return a corrected MDX query that yields one numeric scalar value."
        )

    @staticmethod
    def _looks_like_scalar_question(user_question: str) -> bool:
        q = (user_question or "").lower()
        scalar_signals = [
            "what is",
            "how much",
            "how many",
            "number of",
            "count",
            "total",
            "amount",
            "value",
            "sum",
            "overall",
            "grand total",
        ]
        table_signals = [
            "top",
            "by ",
            "breakdown",
            "per ",
            "list",
            "trend",
            "compare",
        ]
        has_scalar_signal = any(token in q for token in scalar_signals)
        has_table_signal = any(token in q for token in table_signals)
        return has_scalar_signal and not has_table_signal

    @staticmethod
    def _extract_year_from_question(user_question: str) -> str | None:
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", user_question or "")
        if not years:
            return None
        return years[-1]

    @staticmethod
    def _extract_selected_measure_unique_names(selected_objects_text: str) -> list[str]:
        if not selected_objects_text:
            return []
        candidates = re.findall(r"\[Measures\]\.\[[^\]]+\]", selected_objects_text)
        seen: set[str] = set()
        ordered: list[str] = []
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    @staticmethod
    def _is_count_intent(user_question: str) -> bool:
        q = (user_question or "").lower()
        return any(token in q for token in ["how many", "count", "number of"])

    def _measure_exists_in_catalogue(self, measure_unique_name: str) -> bool:
        return measure_unique_name in self._measures_catalogue

    def _choose_measure_for_scalar_question(self, user_question: str, measures: list[str]) -> str | None:
        if not measures:
            return None

        if self._is_count_intent(user_question):
            # Prefer count-style measures when user asks "how many".
            preferred = sorted(
                measures,
                key=lambda m: (
                    "count" not in m.lower(),
                    "internet" not in m.lower(),
                    len(m),
                ),
            )
            return preferred[0]

        # Default behavior for amount/value questions.
        return measures[0]

    def _build_scalar_template_mdx(self, user_question: str, selected_objects_text: str) -> str | None:
        if not self._prefer_scalar_template:
            return None
        if not self._looks_like_scalar_question(user_question):
            return None

        measures = self._extract_selected_measure_unique_names(selected_objects_text)
        if not measures:
            return None

        selected_measure = self._choose_measure_for_scalar_question(user_question, measures)
        if not selected_measure:
            return None
        year = self._extract_year_from_question(user_question)

        if year:
            return (
                "SELECT\n"
                f"{{ {selected_measure} }} ON COLUMNS\n"
                "FROM [Adventure Works]\n"
                f"WHERE ([Date].[Calendar Year].&[{year}])"
            )

        return (
            "SELECT\n"
            f"{{ {selected_measure} }} ON COLUMNS\n"
            "FROM [Adventure Works]"
        )

    @staticmethod
    def _add_timing(timings: dict[str, float], key: str, elapsed_seconds: float) -> None:
        timings[key] = timings.get(key, 0.0) + elapsed_seconds

    def _generate_mdx_with_hybrid_strategy(
        self,
        user_question: str,
        selected_objects_text: str,
        timings: dict[str, float] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        print("[mdx] Using LLM-generated MDX strategy")
        t0 = time.perf_counter()
        mdx_query = self._client.chat(
            messages=[self._build_mdx_prompt(user_question, selected_objects_text)],
            system=_SYSTEM_MDX_QUERY,
        )
        if timings is not None:
            self._add_timing(timings, "mdx_generation_seconds", time.perf_counter() - t0)
        execution = self._answer_from_mdx(user_question, mdx_query, timings=timings)
        return mdx_query, execution

    @staticmethod
    def _normalize_mdx(mdx_text: str) -> str:
        text = (mdx_text or "").strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
        text = text.replace("mdx", "", 1).strip() if text.lower().startswith("mdx") else text
        # Remove invalid empty slicers emitted by repair prompts (e.g. WHERE ()).
        text = re.sub(r"\bWHERE\s*\(\s*\)\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    @staticmethod
    def _extract_first_measure_from_mdx(mdx_text: str) -> str | None:
        match = re.search(r"\[Measures\]\.\[[^\]]+\]", mdx_text or "", flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(0)

    @staticmethod
    def _strip_with_clause(mdx_text: str) -> str:
        text = mdx_text.strip()
        if not text.upper().startswith("WITH"):
            return text

        match = re.search(r"\bSELECT\b", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return text
        return text[match.start():].strip()

    @staticmethod
    def _is_calculated_member_name_collision(error_text: str) -> bool:
        lowered = (error_text or "").lower()
        return (
            "calculated member" in lowered
            and "same name already exists" in lowered
        )

    @staticmethod
    def _is_cube_not_found_error(error_text: str) -> bool:
        lowered = (error_text or "").lower()
        return "cube does not exist" in lowered

    @staticmethod
    def _force_adventure_works_cube(mdx_text: str) -> str:
        # Normalize any cube reference in FROM clause to the known target cube.
        return re.sub(
            r"\bFROM\s+\[[^\]]+\]",
            "FROM [Adventure Works]",
            mdx_text,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _extract_first_numeric_value(rows: list[tuple[Any, ...]]) -> float:
        numeric_token = re.compile(r"[-+]?\d+(?:[\s,]\d{3})*(?:\.\d+)?")

        for row in rows:
            for cell in row:
                if cell is None:
                    continue
                if isinstance(cell, bool):
                    continue
                if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                    value = float(cell)
                    if not (math.isnan(value) or math.isinf(value)):
                        return value

                # Handles CLR numeric wrappers and Decimal-like objects.
                try:
                    value = float(cell)
                    if not (math.isnan(value) or math.isinf(value)):
                        return value
                except Exception:  # noqa: BLE001
                    pass

                # Try common wrapped-value attributes used by .NET/ADOMD objects.
                for attr in ("value", "Value"):
                    wrapped = getattr(cell, attr, None)
                    if wrapped is None:
                        continue
                    try:
                        value = float(wrapped)
                        if not (math.isnan(value) or math.isinf(value)):
                            return value
                    except Exception:  # noqa: BLE001
                        continue

                # Fallback parse from string representation.
                stripped = str(cell).strip().replace(",", "")
                try:
                    value = float(stripped)
                except ValueError:
                    match = numeric_token.search(str(cell))
                    if not match:
                        continue
                    token = match.group(0).replace(" ", "").replace(",", "")
                    try:
                        value = float(token)
                    except ValueError:
                        continue
                if not (math.isnan(value) or math.isinf(value)):
                    return value

        raise ValueError(f"No numeric value found in MDX result set (rows={len(rows)})")

    @staticmethod
    def _preview_rows(rows: list[tuple[Any, ...]], limit: int = 3) -> dict[str, Any]:
        preview_rows = rows[:limit]
        return {
            "row_count": len(rows),
            "preview_values": [[str(cell) for cell in row] for row in preview_rows],
            "preview_types": [[type(cell).__name__ for cell in row] for row in preview_rows],
        }

    @staticmethod
    def _is_effectively_empty_result(rows: list[tuple[Any, ...]]) -> bool:
        if not rows:
            return True
        for row in rows:
            for cell in row:
                if cell is None:
                    continue
                if isinstance(cell, str) and not cell.strip():
                    continue
                return False
        return True

    def _execute_mdx(
        self,
        mdx_query: str,
        timings: dict[str, float] | None = None,
        timing_key: str = "mdx_execution_seconds",
    ) -> tuple[list[tuple[Any, ...]], list[str]]:
        normalized = self._normalize_mdx(mdx_query)
        if not normalized:
            raise ValueError("Generated MDX query is empty")
        t0 = time.perf_counter()
        try:
            rows, headers = self._olap_conn.execute_mdx(normalized)
            if timings is not None:
                self._add_timing(timings, timing_key, time.perf_counter() - t0)
            return rows, list(headers)
        except Exception as exc:  # noqa: BLE001
            if timings is not None:
                self._add_timing(timings, timing_key, time.perf_counter() - t0)
            error_text = str(exc)
            if self._is_calculated_member_name_collision(error_text):
                repaired = self._strip_with_clause(normalized)
                if repaired != normalized:
                    print("[mdx] Retrying query after stripping WITH clause due to calculated-member name collision")
                    t1 = time.perf_counter()
                    rows, headers = self._olap_conn.execute_mdx(repaired)
                    if timings is not None:
                        self._add_timing(timings, timing_key, time.perf_counter() - t1)
                        self._add_timing(timings, "mdx_auto_repair_execution_seconds", time.perf_counter() - t1)
                    return rows, list(headers)
            if self._is_cube_not_found_error(error_text):
                repaired_cube = self._force_adventure_works_cube(normalized)
                if repaired_cube != normalized:
                    print("[mdx] Retrying query after forcing cube name to [Adventure Works]")
                    t2 = time.perf_counter()
                    rows, headers = self._olap_conn.execute_mdx(repaired_cube)
                    if timings is not None:
                        self._add_timing(timings, timing_key, time.perf_counter() - t2)
                        self._add_timing(timings, "mdx_auto_repair_execution_seconds", time.perf_counter() - t2)
                    return rows, list(headers)
            raise

    def _answer_from_mdx(
        self,
        user_question: str,
        mdx_query: str,
        timings: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        used_mdx_query = mdx_query
        try:
            rows, headers = self._execute_mdx(mdx_query, timings=timings, timing_key="mdx_execution_seconds")
        except Exception as exec_exc:  # noqa: BLE001
            print(f"[mdx] Initial execution failed, requesting scalar MDX repair: {exec_exc}")
            t_repair = time.perf_counter()
            repaired_mdx = self._client.chat(
                messages=[self._build_scalar_repair_prompt(user_question, mdx_query)],
                system=_SYSTEM_MDX_SCALAR_REPAIR,
            )
            if timings is not None:
                self._add_timing(timings, "mdx_repair_generation_seconds", time.perf_counter() - t_repair)
            try:
                rows, headers = self._execute_mdx(
                    repaired_mdx,
                    timings=timings,
                    timing_key="mdx_repair_execution_seconds",
                )
            except Exception as repair_exec_exc:  # noqa: BLE001
                print("[mdx] Repair execution failed, returning null result")
                return {
                    "executed_mdx_query": repaired_mdx,
                    "result_headers": [],
                    "result_rows": [],
                    "result_row_count": 0,
                    "singular_value": None,
                    "final_answer": None,
                }
            if used_mdx_query == mdx_query:
                used_mdx_query = repaired_mdx

        try:
            singular_value = self._extract_first_numeric_value(rows)
        except ValueError:
            print("[mdx] No numeric value found, requesting scalar MDX repair")
            t_repair = time.perf_counter()
            repaired_mdx = self._client.chat(
                messages=[self._build_scalar_repair_prompt(user_question, mdx_query)],
                system=_SYSTEM_MDX_SCALAR_REPAIR,
            )
            if timings is not None:
                self._add_timing(timings, "mdx_repair_generation_seconds", time.perf_counter() - t_repair)
            repaired_rows, repaired_headers = self._execute_mdx(
                repaired_mdx,
                timings=timings,
                timing_key="mdx_repair_execution_seconds",
            )
            try:
                singular_value = self._extract_first_numeric_value(repaired_rows)
            except ValueError as repair_exc:
                print("[mdx] Repair execution returned no numeric value, returning null result")
                return {
                    "executed_mdx_query": repaired_mdx,
                    "result_headers": repaired_headers,
                    "result_rows": [list(row) for row in repaired_rows[:10]],
                    "result_row_count": len(repaired_rows),
                    "singular_value": None,
                    "final_answer": None,
                }

            rows = repaired_rows
            headers = repaired_headers
            used_mdx_query = repaired_mdx

        t_answer = time.perf_counter()
        final_answer = self._client.chat(
            messages=[self._build_final_answer_prompt(user_question, singular_value)],
            system=_SYSTEM_FINAL_ANSWER,
        )
        if timings is not None:
            self._add_timing(timings, "final_answer_generation_seconds", time.perf_counter() - t_answer)
        return {
            "executed_mdx_query": used_mdx_query,
            "result_headers": headers,
            "result_rows": [list(row) for row in rows[:10]],
            "result_row_count": len(rows),
            "singular_value": singular_value,
            "final_answer": final_answer,
        }

    # ── public API ────────────────────────────────────────────────────────────

    def run_classic(self, user_question: str) -> dict[str, str]:
        """Run the original two-step catalogue workflow."""
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        print(f"[Step 1] Querying dimensions/levels/attributes for: {user_question!r}")
        t_dims = time.perf_counter()
        dims_response = self._client.chat(
            messages=[self._build_dimensions_prompt(user_question)],
            system=_SYSTEM_DIMENSIONS,
        )
        self._add_timing(timings, "metadata_selection_dimensions_seconds", time.perf_counter() - t_dims)

        print(f"[Step 2] Querying measures for: {user_question!r}")
        t_measures = time.perf_counter()
        measures_response = self._client.chat(
            messages=[self._build_measures_prompt(user_question)],
            system=_SYSTEM_MEASURES,
        )
        self._add_timing(timings, "metadata_selection_measures_seconds", time.perf_counter() - t_measures)
        self._add_timing(
            timings,
            "metadata_selection_total_seconds",
            timings.get("metadata_selection_dimensions_seconds", 0.0)
            + timings.get("metadata_selection_measures_seconds", 0.0),
        )

        selected_objects_text = f"{dims_response}\n\n{measures_response}".strip()
        print("[Classic] Generating/executing MDX with hybrid strategy")
        mdx_query, execution = self._generate_mdx_with_hybrid_strategy(
            user_question,
            selected_objects_text,
            timings=timings,
        )
        self._add_timing(timings, "total_time_seconds", time.perf_counter() - t_total)

        return {
            "workflow": "classic",
            "question": user_question,
            "dimensions_response": dims_response,
            "measures_response": measures_response,
            "mdx_query": mdx_query,
            "timings": timings,
            **execution,
        }

    def run_vector(self, user_question: str) -> dict[str, Any]:
        """Run the FAISS-based metadata workflow over all objects."""
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        vector_store = self._get_vector_store()

        print(f"[Vector Search] Searching all OLAP objects for: {user_question!r}")
        t_vector = time.perf_counter()
        vector_results = vector_store.search(user_question, top_k=self._top_k)
        self._add_timing(timings, "vector_retrieval_seconds", time.perf_counter() - t_vector)
        top_objects = [result.payload for result in vector_results]

        print(f"[Vector Search] Retrieved top {len(top_objects)} objects, sending full JSON to chatbot")
        t_meta = time.perf_counter()
        metadata_response = self._client.chat(
            messages=[self._build_vector_prompt(user_question, top_objects)],
            system=_SYSTEM_VECTOR_METADATA,
        )
        self._add_timing(timings, "metadata_selection_total_seconds", time.perf_counter() - t_meta)

        print("[Vector Search] Generating/executing MDX with hybrid strategy")
        mdx_query, execution = self._generate_mdx_with_hybrid_strategy(
            user_question,
            metadata_response,
            timings=timings,
        )
        self._add_timing(timings, "total_time_seconds", time.perf_counter() - t_total)

        return {
            "workflow": "vector",
            "question": user_question,
            "top_k": self._top_k,
            "vector_results": self._serialize_vector_results(vector_results),
            "top_objects": top_objects,
            "metadata_response": metadata_response,
            "mdx_query": mdx_query,
            "timings": timings,
            **execution,
        }

    def run_vector_hierarchical(self, user_question: str) -> dict[str, Any]:
        """Run a hierarchical FAISS workflow: dimensions -> children within dimensions."""
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        vector_store = self._get_vector_store()

        print(f"[Vector Hierarchical] Searching top dimensions for: {user_question!r}")
        t_dim = time.perf_counter()
        top_dimensions = vector_store.search_dimensions(
            user_question,
            top_k=self._dimension_top_k,
        )
        self._add_timing(timings, "vector_retrieval_dimensions_seconds", time.perf_counter() - t_dim)
        selected_dimension_unique_names = [
            result.unique_name for result in top_dimensions if result.unique_name
        ]

        print(
            "[Vector Hierarchical] Searching child metadata within selected dimensions "
            f"(top {self._child_top_k_per_dimension} per dimension)"
        )
        t_child = time.perf_counter()
        top_children = vector_store.search_children_in_dimensions(
            user_question,
            dimension_unique_names=selected_dimension_unique_names,
            top_k_per_dimension=self._child_top_k_per_dimension,
        )
        self._add_timing(timings, "vector_retrieval_children_seconds", time.perf_counter() - t_child)

        print(f"[Vector Hierarchical] Searching top measures for: {user_question!r}")
        t_meas = time.perf_counter()
        top_measures = vector_store.search_measures(
            user_question,
            top_k=self._measure_top_k,
        )
        self._add_timing(timings, "vector_retrieval_measures_seconds", time.perf_counter() - t_meas)
        self._add_timing(
            timings,
            "vector_retrieval_seconds",
            timings.get("vector_retrieval_dimensions_seconds", 0.0)
            + timings.get("vector_retrieval_children_seconds", 0.0)
            + timings.get("vector_retrieval_measures_seconds", 0.0),
        )

        top_objects = [result.payload for result in top_dimensions]
        top_objects.extend(result.payload for result in top_children)
        top_objects.extend(result.payload for result in top_measures)

        print(
            "[Vector Hierarchical] Retrieved "
            f"{len(top_dimensions)} dimensions, {len(top_children)} children, {len(top_measures)} measures"
        )
        t_meta = time.perf_counter()
        metadata_response = self._client.chat(
            messages=[self._build_vector_prompt(user_question, top_objects)],
            system=_SYSTEM_VECTOR_METADATA,
        )
        self._add_timing(timings, "metadata_selection_total_seconds", time.perf_counter() - t_meta)

        print("[Vector Hierarchical] Generating/executing MDX with hybrid strategy")
        mdx_query, execution = self._generate_mdx_with_hybrid_strategy(
            user_question,
            metadata_response,
            timings=timings,
        )
        self._add_timing(timings, "total_time_seconds", time.perf_counter() - t_total)

        return {
            "workflow": "vector_hierarchical",
            "question": user_question,
            "dimension_top_k": self._dimension_top_k,
            "child_top_k_per_dimension": self._child_top_k_per_dimension,
            "measure_top_k": self._measure_top_k,
            "top_dimensions": self._serialize_vector_results(top_dimensions),
            "top_children": self._serialize_vector_results(top_children),
            "top_measures": self._serialize_vector_results(top_measures),
            "top_objects": top_objects,
            "metadata_response": metadata_response,
            "mdx_query": mdx_query,
            "timings": timings,
            **execution,
        }

    def run_vector_two_tier(self, user_question: str) -> dict[str, Any]:
        """Run vector search with two tiers: dimensions+children, measures."""
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        vector_store = self._get_vector_store()

        print(f"[Vector Two-Tier] Searching dimensions and children for: {user_question!r}")
        t_dim_child = time.perf_counter()
        top_dim_child = vector_store.search_dimensions_and_children(
            user_question,
            top_k=self._dimension_top_k + self._child_top_k_per_dimension,
        )
        self._add_timing(timings, "vector_retrieval_dimensions_and_children_seconds", time.perf_counter() - t_dim_child)

        print(f"[Vector Two-Tier] Searching top measures for: {user_question!r}")
        t_meas = time.perf_counter()
        top_measures = vector_store.search_measures(
            user_question,
            top_k=self._measure_top_k,
        )
        self._add_timing(timings, "vector_retrieval_measures_seconds", time.perf_counter() - t_meas)
        self._add_timing(
            timings,
            "vector_retrieval_seconds",
            timings.get("vector_retrieval_dimensions_and_children_seconds", 0.0)
            + timings.get("vector_retrieval_measures_seconds", 0.0),
        )

        top_objects = [result.payload for result in top_dim_child]
        top_objects.extend(result.payload for result in top_measures)

        print(
            "[Vector Two-Tier] Retrieved "
            f"{len(top_dim_child)} dimensions/children, {len(top_measures)} measures"
        )
        t_meta = time.perf_counter()
        metadata_response = self._client.chat(
            messages=[self._build_vector_prompt(user_question, top_objects)],
            system=_SYSTEM_VECTOR_METADATA,
        )
        self._add_timing(timings, "metadata_selection_total_seconds", time.perf_counter() - t_meta)

        print("[Vector Two-Tier] Generating/executing MDX with hybrid strategy")
        mdx_query, execution = self._generate_mdx_with_hybrid_strategy(
            user_question,
            metadata_response,
            timings=timings,
        )
        self._add_timing(timings, "total_time_seconds", time.perf_counter() - t_total)

        return {
            "workflow": "vector_two_tier",
            "question": user_question,
            "top_dim_child": self._serialize_vector_results(top_dim_child),
            "top_measures": self._serialize_vector_results(top_measures),
            "top_objects": top_objects,
            "metadata_response": metadata_response,
            "mdx_query": mdx_query,
            "timings": timings,
            **execution,
        }

    def run(self, user_question: str) -> dict[str, Any]:
        """Run the configured metadata-selection workflow."""
        if self._workflow == "vector":
            return self.run_vector(user_question)
        if self._workflow == "vector_hierarchical":
            return self.run_vector_hierarchical(user_question)
        if self._workflow == "vector_two_tier":
            return self.run_vector_two_tier(user_question)
        return self.run_classic(user_question)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Show me the top 5 product subcategories by internet sales amount for female customers in North America, broken down by calendar quarter for fiscal year 2023."

    builder = OLAPPromptBuilder()
    result = builder.run(question)
    if result.get("workflow") == "vector":
        print("\n" + "=" * 60)
        print("VECTOR WORKFLOW – Top Retrieved Objects:")
        print("=" * 60)
        for item in result["vector_results"]:
            print(
                f"#{item['rank']:02d} score={item['score']:.4f} "
                f"type={item['object_type']} name={item['name']} unique_name={item['unique_name']}"
            )

        print("\n" + "=" * 60)
        print("VECTOR WORKFLOW – Chatbot Selection:")
        print("=" * 60)
        print(result["metadata_response"])

        print("\n" + "=" * 60)
        print("VECTOR WORKFLOW – Final MDX Query:")
        print("=" * 60)
        print(result.get("mdx_query", ""))

        print("\n" + "=" * 60)
        print("VECTOR WORKFLOW – Executed MDX Query:")
        print("=" * 60)
        print(result.get("executed_mdx_query", ""))

        print("\n" + "=" * 60)
        print("VECTOR WORKFLOW – Singular Value Answer:")
        print("=" * 60)
        print(f"value={result.get('singular_value')}")
        print(result.get("final_answer", ""))
    elif result.get("workflow") == "vector_hierarchical":
        print("\n" + "=" * 60)
        print("VECTOR HIERARCHICAL – Top Dimensions:")
        print("=" * 60)
        for item in result["top_dimensions"]:
            print(
                f"#{item['rank']:02d} score={item['score']:.4f} "
                f"type={item['object_type']} name={item['name']} unique_name={item['unique_name']}"
            )

        print("\n" + "=" * 60)
        print("VECTOR HIERARCHICAL – Top Children:")
        print("=" * 60)
        for item in result["top_children"]:
            print(
                f"#{item['rank']:02d} score={item['score']:.4f} "
                f"type={item['object_type']} name={item['name']} unique_name={item['unique_name']}"
            )

        print("\n" + "=" * 60)
        print("VECTOR HIERARCHICAL – Top Measures:")
        print("=" * 60)
        for item in result["top_measures"]:
            print(
                f"#{item['rank']:02d} score={item['score']:.4f} "
                f"type={item['object_type']} name={item['name']} unique_name={item['unique_name']}"
            )

        print("\n" + "=" * 60)
        print("VECTOR HIERARCHICAL – Chatbot Selection:")
        print("=" * 60)
        print(result["metadata_response"])

        print("\n" + "=" * 60)
        print("VECTOR HIERARCHICAL – Final MDX Query:")
        print("=" * 60)
        print(result.get("mdx_query", ""))

        print("\n" + "=" * 60)
        print("VECTOR HIERARCHICAL – Executed MDX Query:")
        print("=" * 60)
        print(result.get("executed_mdx_query", ""))

        print("\n" + "=" * 60)
        print("VECTOR HIERARCHICAL – Singular Value Answer:")
        print("=" * 60)
        print(f"value={result.get('singular_value')}")
        print(result.get("final_answer", ""))
    elif result.get("workflow") == "vector_two_tier":
        print("\n" + "=" * 60)
        print("VECTOR TWO-TIER – Top Dimensions/Children:")
        print("=" * 60)
        for item in result["top_dim_child"]:
            print(
                f"#{item['rank']:02d} score={item['score']:.4f} "
                f"type={item['object_type']} name={item['name']} unique_name={item['unique_name']}"
            )

        print("\n" + "=" * 60)
        print("VECTOR TWO-TIER – Top Measures:")
        print("=" * 60)
        for item in result["top_measures"]:
            print(
                f"#{item['rank']:02d} score={item['score']:.4f} "
                f"type={item['object_type']} name={item['name']} unique_name={item['unique_name']}"
            )

        print("\n" + "=" * 60)
        print("VECTOR TWO-TIER – Chatbot Selection:")
        print("=" * 60)
        print(result["metadata_response"])

        print("\n" + "=" * 60)
        print("VECTOR TWO-TIER – Final MDX Query:")
        print("=" * 60)
        print(result.get("mdx_query", ""))

        print("\n" + "=" * 60)
        print("VECTOR TWO-TIER – Executed MDX Query:")
        print("=" * 60)
        print(result.get("executed_mdx_query", ""))

        print("\n" + "=" * 60)
        print("VECTOR TWO-TIER – Singular Value Answer:")
        print("=" * 60)
        print(f"value={result.get('singular_value')}")
        print(result.get("final_answer", ""))
    else:
        print("\n" + "=" * 60)
        print("STEP 1 – Relevant dimensions / levels / attributes:")
        print("=" * 60)
        print(result["dimensions_response"])

        print("\n" + "=" * 60)
        print("STEP 2 – Relevant measures:")
        print("=" * 60)
        print(result["measures_response"])

        print("\n" + "=" * 60)
        print("CLASSIC WORKFLOW – Final MDX Query:")
        print("=" * 60)
        print(result.get("mdx_query", ""))

        print("\n" + "=" * 60)
        print("CLASSIC WORKFLOW – Executed MDX Query:")
        print("=" * 60)
        print(result.get("executed_mdx_query", ""))

        print("\n" + "=" * 60)
        print("CLASSIC WORKFLOW – Singular Value Answer:")
        print("=" * 60)
        print(f"value={result.get('singular_value')}")
        print(result.get("final_answer", ""))
