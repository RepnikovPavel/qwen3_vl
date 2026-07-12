from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Sequence


class SchemaError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise SchemaError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads_json(text: str, source: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except SchemaError:
        raise
    except json.JSONDecodeError as exc:
        raise SchemaError(
            f"invalid JSON in {source}: {exc.msg} at line {exc.lineno}"
        ) from exc


def _load_json(path: Path) -> tuple[Any, str]:
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"JSON is not UTF-8: {path}") from exc
    return _loads_json(text, str(path)), hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{path} must be an object")
    return value


def _require_exact_keys(value: Any, keys: set[str], path: str) -> dict[str, Any]:
    mapping = _require_mapping(value, path)
    actual = set(mapping)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise SchemaError(f"{path} keys mismatch: missing={missing}, extra={extra}")
    return mapping


def _require_string(value: Any, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise SchemaError(f"{path} must be {qualifier}")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(f"{path} must be an array")
    return value


def _require_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise SchemaError(f"{path} must be a finite number")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latex_tokens(value: str) -> list[str]:
    return re.findall(
        r"\\[A-Za-z]+|\\.|[A-Za-z]+|\d+(?:\.\d+)?|\{|\}|\[|\]|[_^=+\-*/(),]|[^\s]",
        value,
    )


def _latex_syntax_like(value: str) -> bool:
    if not value.strip() or "\x00" in value:
        return False
    stack: list[str] = []
    pairs = {"}": "{", "]": "[", ")": "("}
    dollar_count = 0
    for token in _latex_tokens(value):
        if token.startswith("\\"):
            continue
        if token == "$":
            dollar_count += 1
        elif token in ("{", "[", "("):
            stack.append(token)
        elif token in pairs:
            if not stack or stack.pop() != pairs[token]:
                return False
    return not stack and dollar_count % 2 == 0 and not value.rstrip().endswith("\\")


def _validate_formula_object(
    value: Any, path: str, require_nonempty: bool
) -> dict[str, Any]:
    formula_object = _require_exact_keys(value, {"formulas"}, path)
    formulas = _require_list(formula_object["formulas"], f"{path}.formulas")
    if require_nonempty and not formulas:
        raise SchemaError(f"{path}.formulas must not be empty")
    for index, formula in enumerate(formulas):
        _require_string(
            formula, f"{path}.formulas[{index}]", allow_empty=not require_nonempty
        )
    return formula_object


def _validate_fact(value: Any, path: str) -> dict[str, Any]:
    fact = _require_exact_keys(value, {"subject", "relation", "object"}, path)
    for key in ("subject", "relation", "object"):
        _require_string(fact[key], f"{path}.{key}")
    return fact


def _validate_series(value: Any, path: str) -> dict[str, Any]:
    series = _require_exact_keys(value, {"name", "values"}, path)
    _require_string(series["name"], f"{path}.name")
    values = _require_list(series["values"], f"{path}.values")
    for index, number in enumerate(values):
        _require_number(number, f"{path}.values[{index}]")
    return series


def _validate_panel(value: Any, path: str) -> dict[str, Any]:
    panel = _require_exact_keys(
        value,
        {"chart_type", "title", "x_label", "y_label", "categories", "series"},
        path,
    )
    chart_type = _require_string(panel["chart_type"], f"{path}.chart_type")
    if chart_type not in {"line", "bar", "scatter", "heatmap"}:
        raise SchemaError(f"{path}.chart_type is unsupported: {chart_type}")
    for key in ("title", "x_label", "y_label"):
        _require_string(panel[key], f"{path}.{key}")
    categories = _require_list(panel["categories"], f"{path}.categories")
    for index, category in enumerate(categories):
        _require_string(category, f"{path}.categories[{index}]")
    series = _require_list(panel["series"], f"{path}.series")
    for index, item in enumerate(series):
        _validate_series(item, f"{path}.series[{index}]")
    return panel


def _validate_chart_object(
    value: Any, path: str, include_tolerance: bool
) -> dict[str, Any]:
    keys = {"title", "panels", "facts"}
    if include_tolerance:
        keys.add("numeric_tolerance")
    chart = _require_exact_keys(value, keys, path)
    _require_string(chart["title"], f"{path}.title")
    panels = _require_list(chart["panels"], f"{path}.panels")
    for index, panel in enumerate(panels):
        _validate_panel(panel, f"{path}.panels[{index}]")
    facts = _require_list(chart["facts"], f"{path}.facts")
    for index, fact in enumerate(facts):
        _validate_fact(fact, f"{path}.facts[{index}]")
    if include_tolerance:
        tolerance = _require_number(
            chart["numeric_tolerance"], f"{path}.numeric_tolerance"
        )
        if tolerance < 0:
            raise SchemaError(f"{path}.numeric_tolerance must be non-negative")
    return chart


def validate_manifest(
    data: Any, directory: Path, verify_images: bool = True
) -> list[dict[str, Any]]:
    manifest = _require_exact_keys(data, {"schema_version", "fixtures"}, "manifest")
    if manifest["schema_version"] != 1:
        raise SchemaError("manifest.schema_version must equal 1")
    fixtures = _require_list(manifest["fixtures"], "manifest.fixtures")
    if len(fixtures) != 4:
        raise SchemaError("manifest.fixtures must contain exactly four fixtures")
    seen: set[str] = set()
    task_counts = {"text": 0, "formula": 0, "chart": 0}
    validated: list[dict[str, Any]] = []
    for index, fixture_value in enumerate(fixtures):
        path = f"manifest.fixtures[{index}]"
        fixture = _require_exact_keys(
            fixture_value,
            {"id", "task", "image", "image_sha256", "prompt", "ground_truth"},
            path,
        )
        fixture_id = _require_string(fixture["id"], f"{path}.id")
        if not re.fullmatch(r"[a-z0-9_]+", fixture_id):
            raise SchemaError(f"{path}.id must match [a-z0-9_]+")
        if fixture_id in seen:
            raise SchemaError(f"duplicate fixture id: {fixture_id}")
        seen.add(fixture_id)
        task = _require_string(fixture["task"], f"{path}.task")
        if task not in task_counts:
            raise SchemaError(f"{path}.task is unsupported: {task}")
        task_counts[task] += 1
        image_name = _require_string(fixture["image"], f"{path}.image")
        image_relative = Path(image_name)
        if (
            image_relative.is_absolute()
            or image_relative.name != image_name
            or image_relative.suffix != ".png"
        ):
            raise SchemaError(f"{path}.image must be a safe PNG basename")
        image_digest = _require_string(fixture["image_sha256"], f"{path}.image_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", image_digest):
            raise SchemaError(f"{path}.image_sha256 must be lowercase SHA-256")
        _require_string(fixture["prompt"], f"{path}.prompt")
        ground_truth_path = f"{path}.ground_truth"
        if task == "text":
            ground_truth = _require_exact_keys(
                fixture["ground_truth"], {"text"}, ground_truth_path
            )
            _require_string(ground_truth["text"], f"{ground_truth_path}.text")
        elif task == "formula":
            _validate_formula_object(fixture["ground_truth"], ground_truth_path, True)
        else:
            _validate_chart_object(fixture["ground_truth"], ground_truth_path, True)
        if verify_images:
            image_path = directory / image_name
            if not image_path.is_file():
                raise SchemaError(f"fixture image is missing: {image_name}")
            if _sha256(image_path) != image_digest:
                raise SchemaError(f"fixture image SHA-256 mismatch: {image_name}")
        validated.append(fixture)
    if task_counts != {"text": 1, "formula": 1, "chart": 2}:
        raise SchemaError(f"manifest task distribution is invalid: {task_counts}")
    return validated


def validate_responses(data: Any, fixture_ids: set[str]) -> dict[str, str]:
    wrapper = _require_exact_keys(data, {"schema_version", "responses"}, "responses")
    if wrapper["schema_version"] != 1:
        raise SchemaError("responses.schema_version must equal 1")
    entries = _require_list(wrapper["responses"], "responses.responses")
    result: dict[str, str] = {}
    for index, entry_value in enumerate(entries):
        path = f"responses.responses[{index}]"
        entry = _require_exact_keys(entry_value, {"id", "answer"}, path)
        fixture_id = _require_string(entry["id"], f"{path}.id")
        answer = _require_string(entry["answer"], f"{path}.answer", allow_empty=True)
        if fixture_id in result:
            raise SchemaError(f"duplicate response id: {fixture_id}")
        result[fixture_id] = answer
    actual = set(result)
    if actual != fixture_ids:
        raise SchemaError(
            f"response ids mismatch: missing={sorted(fixture_ids - actual)}, extra={sorted(actual - fixture_ids)}"
        )
    return result


def _levenshtein(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _nfkc(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def _normalized_label(value: str) -> str:
    return " ".join(_nfkc(value).casefold().split())


def _normalized_latex(value: str) -> str:
    result = _nfkc(value).strip()
    if result.startswith(r"\(") and result.endswith(r"\)"):
        result = result[2:-2]
    if len(result) >= 2 and result.startswith("$") and result.endswith("$"):
        result = result[1:-1]
    result = result.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    result = result.replace(r"\left", "").replace(r"\right", "")
    return re.sub(r"\s+", "", result)


def _similarity(left: Sequence[Any], right: Sequence[Any]) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return 1.0 - _levenshtein(left, right) / denominator


def _evaluate_text(reference: str, prediction: str) -> dict[str, Any]:
    normalized_reference = _nfkc(reference)
    normalized_prediction = _nfkc(prediction)
    reference_words = normalized_reference.split()
    prediction_words = normalized_prediction.split()
    character_distance = _levenshtein(normalized_reference, normalized_prediction)
    word_distance = _levenshtein(reference_words, prediction_words)
    return {
        "nfkc_exact": normalized_reference == normalized_prediction,
        "reference_characters": len(normalized_reference),
        "predicted_characters": len(normalized_prediction),
        "character_edit_distance": character_distance,
        "cer": character_distance / max(len(normalized_reference), 1),
        "reference_words": len(reference_words),
        "predicted_words": len(prediction_words),
        "word_edit_distance": word_distance,
        "wer": word_distance / max(len(reference_words), 1),
    }


def _evaluate_formula(reference: list[str], prediction: list[str]) -> dict[str, Any]:
    slot_count = max(len(reference), len(prediction), 1)
    exact_count = 0
    token_similarity_sum = 0.0
    syntax_valid_count = 0
    for index in range(slot_count):
        expected = reference[index] if index < len(reference) else None
        actual = prediction[index] if index < len(prediction) else None
        if expected is not None and actual is not None:
            normalized_expected = _normalized_latex(expected)
            normalized_actual = _normalized_latex(actual)
            exact_count += normalized_expected == normalized_actual
            token_similarity_sum += _similarity(
                _latex_tokens(normalized_expected),
                _latex_tokens(normalized_actual),
            )
        if actual is not None and _latex_syntax_like(actual):
            syntax_valid_count += 1
    return {
        "expected_count": len(reference),
        "predicted_count": len(prediction),
        "normalized_exact_count": exact_count,
        "normalized_exact_rate": exact_count / slot_count,
        "token_edit_similarity_mean": token_similarity_sum / slot_count,
        "syntax_valid_count": syntax_valid_count,
        "syntax_valid_rate": syntax_valid_count / slot_count,
    }


def _chart_labels(chart: dict[str, Any]) -> list[str]:
    labels = [chart["title"]]
    for panel in chart["panels"]:
        labels.extend((panel["title"], panel["x_label"], panel["y_label"]))
        labels.extend(panel["categories"])
        labels.extend(series["name"] for series in panel["series"])
    return labels


def _chart_types(chart: dict[str, Any]) -> list[str]:
    return [panel["chart_type"] for panel in chart["panels"]]


def _chart_numbers(chart: dict[str, Any]) -> list[float]:
    return [
        float(value)
        for panel in chart["panels"]
        for series in panel["series"]
        for value in series["values"]
    ]


def _chart_facts(chart: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (
            _normalized_label(fact["subject"]),
            _normalized_label(fact["relation"]),
            _normalized_label(fact["object"]),
        )
        for fact in chart["facts"]
    }


def _evaluate_chart(
    reference: dict[str, Any], prediction: dict[str, Any] | None
) -> dict[str, Any]:
    expected_labels = _chart_labels(reference)
    predicted_labels = _chart_labels(prediction) if prediction is not None else []
    label_slots = max(len(expected_labels), len(predicted_labels), 1)
    label_exact_count = 0
    label_similarity_sum = 0.0
    for index in range(label_slots):
        expected = (
            _normalized_label(expected_labels[index])
            if index < len(expected_labels)
            else None
        )
        actual = (
            _normalized_label(predicted_labels[index])
            if index < len(predicted_labels)
            else None
        )
        if expected is not None and actual is not None:
            label_exact_count += expected == actual
            label_similarity_sum += _similarity(expected, actual)
    expected_types = _chart_types(reference)
    predicted_types = _chart_types(prediction) if prediction is not None else []
    type_slots = max(len(expected_types), len(predicted_types), 1)
    type_exact_count = sum(
        index < len(expected_types)
        and index < len(predicted_types)
        and expected_types[index] == predicted_types[index]
        for index in range(type_slots)
    )
    expected_numbers = _chart_numbers(reference)
    predicted_numbers = _chart_numbers(prediction) if prediction is not None else []
    number_slots = max(len(expected_numbers), len(predicted_numbers), 1)
    aligned_errors = [
        abs(expected_numbers[index] - predicted_numbers[index])
        for index in range(min(len(expected_numbers), len(predicted_numbers)))
    ]
    tolerance = float(reference["numeric_tolerance"])
    within_tolerance = sum(error <= tolerance for error in aligned_errors)
    expected_facts = _chart_facts(reference)
    predicted_facts = _chart_facts(prediction) if prediction is not None else set()
    fact_matches = len(expected_facts & predicted_facts)
    precision = fact_matches / len(predicted_facts) if predicted_facts else 0.0
    recall = fact_matches / len(expected_facts) if expected_facts else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "expected_panel_count": len(reference["panels"]),
        "predicted_panel_count": len(prediction["panels"])
        if prediction is not None
        else 0,
        "chart_type_exact_count": type_exact_count,
        "chart_type_exact_rate": type_exact_count / type_slots,
        "expected_label_count": len(expected_labels),
        "predicted_label_count": len(predicted_labels),
        "label_exact_count": label_exact_count,
        "label_exact_rate": label_exact_count / label_slots,
        "label_edit_similarity_mean": label_similarity_sum / label_slots,
        "expected_numeric_count": len(expected_numbers),
        "predicted_numeric_count": len(predicted_numbers),
        "numeric_aligned_count": len(aligned_errors),
        "numeric_mae_aligned": sum(aligned_errors) / len(aligned_errors)
        if aligned_errors
        else None,
        "numeric_tolerance": tolerance,
        "numeric_within_tolerance_count": within_tolerance,
        "numeric_within_tolerance_rate": within_tolerance / number_slots,
        "expected_fact_count": len(expected_facts),
        "predicted_fact_count": len(predicted_facts),
        "fact_match_count": fact_matches,
        "fact_precision": precision,
        "fact_recall": recall,
        "fact_f1": f1,
    }


def _structured_answer(
    answer: str, task: str
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = _loads_json(answer, f"{task} answer")
        if task == "formula":
            return _validate_formula_object(parsed, "answer", False), None
        return _validate_chart_object(parsed, "answer", False), None
    except SchemaError as exc:
        return None, str(exc)


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    schema_valid_count = sum(result["response_schema_valid"] for result in results)
    text_result = next(result for result in results if result["task"] == "text")
    formula_result = next(result for result in results if result["task"] == "formula")
    chart_results = [result for result in results if result["task"] == "chart"]
    chart_metrics = [result["metrics"] for result in chart_results]
    label_slots = sum(
        max(metric["expected_label_count"], metric["predicted_label_count"], 1)
        for metric in chart_metrics
    )
    label_matches = sum(metric["label_exact_count"] for metric in chart_metrics)
    type_slots = sum(
        max(metric["expected_panel_count"], metric["predicted_panel_count"], 1)
        for metric in chart_metrics
    )
    type_matches = sum(metric["chart_type_exact_count"] for metric in chart_metrics)
    number_slots = sum(
        max(metric["expected_numeric_count"], metric["predicted_numeric_count"], 1)
        for metric in chart_metrics
    )
    numeric_matches = sum(
        metric["numeric_within_tolerance_count"] for metric in chart_metrics
    )
    expected_facts = sum(metric["expected_fact_count"] for metric in chart_metrics)
    fact_matches = sum(metric["fact_match_count"] for metric in chart_metrics)
    predicted_facts = sum(metric["predicted_fact_count"] for metric in chart_metrics)
    fact_precision = fact_matches / predicted_facts if predicted_facts else 0.0
    fact_recall = fact_matches / expected_facts if expected_facts else 1.0
    return {
        "fixture_count": len(results),
        "response_schema_valid_count": schema_valid_count,
        "response_schema_valid_rate": schema_valid_count / len(results),
        "text": {
            "nfkc_exact": text_result["metrics"]["nfkc_exact"],
            "cer": text_result["metrics"]["cer"],
            "wer": text_result["metrics"]["wer"],
        },
        "formula": {
            "normalized_exact_rate": formula_result["metrics"]["normalized_exact_rate"],
            "token_edit_similarity_mean": formula_result["metrics"][
                "token_edit_similarity_mean"
            ],
            "syntax_valid_rate": formula_result["metrics"]["syntax_valid_rate"],
        },
        "charts": {
            "chart_type_exact_rate": type_matches / type_slots,
            "label_exact_rate": label_matches / label_slots,
            "numeric_within_tolerance_rate": numeric_matches / number_slots,
            "fact_precision": fact_precision,
            "fact_recall": fact_recall,
            "fact_f1": (
                2 * fact_precision * fact_recall / (fact_precision + fact_recall)
                if fact_precision + fact_recall
                else 0.0
            ),
        },
    }


def evaluate(manifest_path: str | Path, responses_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    responses_file = Path(responses_path).expanduser().resolve()
    manifest_data, manifest_digest = _load_json(manifest_file)
    fixtures = validate_manifest(
        manifest_data, manifest_file.parent, verify_images=True
    )
    responses_data, responses_digest = _load_json(responses_file)
    answers = validate_responses(
        responses_data, {fixture["id"] for fixture in fixtures}
    )
    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        answer = answers[fixture["id"]]
        task = fixture["task"]
        schema_error = None
        if task == "text":
            schema_valid = True
            metrics = _evaluate_text(fixture["ground_truth"]["text"], answer)
        elif task == "formula":
            structured, schema_error = _structured_answer(answer, task)
            schema_valid = structured is not None
            predicted = structured["formulas"] if structured is not None else []
            metrics = _evaluate_formula(fixture["ground_truth"]["formulas"], predicted)
        else:
            structured, schema_error = _structured_answer(answer, task)
            schema_valid = structured is not None
            metrics = _evaluate_chart(fixture["ground_truth"], structured)
        results.append(
            {
                "id": fixture["id"],
                "task": task,
                "response_schema_valid": schema_valid,
                "response_schema_error": schema_error,
                "metrics": metrics,
            }
        )
    return {
        "schema_version": 1,
        "manifest_sha256": manifest_digest,
        "responses_sha256": responses_digest,
        "results": results,
        "summary": _summary(results),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        result = evaluate(args.manifest, args.responses)
    except (OSError, SchemaError) as exc:
        print(
            json.dumps({"schema_version": 1, "ok": False, "error": str(exc)}),
            file=sys.stderr,
        )
        return 2
    if args.output:
        _write_json(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
