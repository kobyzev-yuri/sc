import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent

BASE_PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
SCALE_PREDICTIONS_DIR = PROJECT_ROOT / "scale_results" / "predictions"
MERGED_PREDICTIONS_DIR = PROJECT_ROOT / "merged_results" / "predictions"

BASE_INFERENCE_DIR = PROJECT_ROOT / "results" / "inference"
SCALE_INFERENCE_DIR = PROJECT_ROOT / "scale_results" / "inference"
MERGED_INFERENCE_DIR = PROJECT_ROOT / "merged_results" / "inference"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use compact separators but keep utf-8
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two prediction dicts.

    - Keys unique to either dict are kept as is.
    - For intersecting keys, if both values are lists (as for detection lists),
      they are concatenated.
    - Otherwise, the value from `b` overrides `a`.
    """
    result: Dict[str, Any] = dict(a)

    for key, b_val in b.items():
        if key in result:
            a_val = result[key]
            if isinstance(a_val, list) and isinstance(b_val, list):
                result[key] = a_val + b_val
            else:
                # If structure is different, prefer scale_results version
                result[key] = b_val
        else:
            result[key] = b_val

    return result


def merge_dir_pair(base_dir: Path, scale_dir: Path, out_dir: Path, label: str) -> None:
    if not base_dir.is_dir():
        raise SystemExit(f"Base {label} directory not found: {base_dir}")
    if not scale_dir.is_dir():
        raise SystemExit(f"Scale {label} directory not found: {scale_dir}")

    base_files = {p.name: p for p in base_dir.glob("*.json")}
    scale_files = {p.name: p for p in scale_dir.glob("*.json")}

    common_names = sorted(set(base_files.keys()) & set(scale_files.keys()))

    if not common_names:
        print(f"No common JSON filenames found between {base_dir} and {scale_dir}")
        return

    print(f"[{label}] Found {len(common_names)} common JSON files to merge.")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in common_names:
        base_path = base_files[name]
        scale_path = scale_files[name]

        print(f"[{label}] Merging {name}")
        base_data = load_json(base_path)
        scale_data = load_json(scale_path)

        if not isinstance(base_data, dict) or not isinstance(scale_data, dict):
            raise ValueError(
                f"[{label}] Expected dict JSON at {name}, got {type(base_data)} and {type(scale_data)}"
            )

        merged = merge_dicts(base_data, scale_data)
        out_path = out_dir / name
        save_json(merged, out_path)


def main() -> None:
    # Merge predictions
    merge_dir_pair(
        BASE_PREDICTIONS_DIR,
        SCALE_PREDICTIONS_DIR,
        MERGED_PREDICTIONS_DIR,
        label="predictions",
    )

    # Merge inference
    merge_dir_pair(
        BASE_INFERENCE_DIR,
        SCALE_INFERENCE_DIR,
        MERGED_INFERENCE_DIR,
        label="inference",
    )

    print(f"Done. Merged predictions -> {MERGED_PREDICTIONS_DIR}")
    print(f"Done. Merged inference   -> {MERGED_INFERENCE_DIR}")


if __name__ == "__main__":
    main()


