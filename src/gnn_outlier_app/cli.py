from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import load_table, prepare_features
from .database import PERSIST_ALL, PERSIST_OUTLIERS, get_engine, save_run
from .explain import top_feature_contributions
from .export import attach_results, to_csv_bytes, to_excel_bytes, to_parquet_bytes
from .graph import build_knn_graph
from .models import ARCHITECTURE_DESCRIPTIONS, THRESHOLD_STRATEGIES, run_detection

OUTPUT_WRITERS = {
    ".csv": to_csv_bytes,
    ".xlsx": to_excel_bytes,
    ".parquet": to_parquet_bytes,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gnn-outlier-studio",
        description="Headless scoring pipeline for GNN-based tabular outlier detection.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="Score a tabular file and write results.")
    score.add_argument("input", help="Path to a .txt, .csv, .xls, .xlsx, or .parquet file.")
    score.add_argument("--out", required=True, help="Output path (.csv, .xlsx, or .parquet).")
    score.add_argument("--k", type=int, default=8, help="kNN graph neighbors (default: 8).")
    score.add_argument("--contamination", type=float, default=0.05, help="Expected outlier rate (default: 0.05).")
    score.add_argument("--epochs", type=int, default=80, help="Training epochs (default: 80).")
    score.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension (default: 32).")
    score.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    score.add_argument(
        "--architectures",
        default=",".join(ARCHITECTURE_DESCRIPTIONS),
        help="Comma-separated GNN architectures (default: all).",
    )
    score.add_argument(
        "--threshold-strategy",
        choices=THRESHOLD_STRATEGIES,
        default="contamination",
        help="'contamination' flags the top quantile; 'mad' may flag nothing on clean data.",
    )
    score.add_argument("--outliers-only", action="store_true", help="Write only flagged rows.")
    score.add_argument("--top-features", type=int, default=3, help="Explanation features per row (default: 3).")
    score.add_argument("--db", default=None, help="Optional SQLite path to persist the run.")
    score.add_argument("--persist-all", action="store_true", help="Persist all rows to the database, not just outliers.")
    return parser


def _print_progress(label: str, completed: int, total: int) -> None:
    print(f"  trained {label} ({completed}/{total})", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_path = Path(args.out)
    writer = OUTPUT_WRITERS.get(out_path.suffix.lower())
    if writer is None:
        print(f"error: unsupported output format '{out_path.suffix}'. Use .csv, .xlsx, or .parquet.", file=sys.stderr)
        return 2

    architectures = [name.strip() for name in args.architectures.split(",") if name.strip()]

    try:
        loaded = load_table(args.input)
        prepared = prepare_features(loaded.frame)
        if not prepared.feature_columns:
            print("error: automatic preprocessing found no usable model features.", file=sys.stderr)
            return 2
        for note in prepared.notes:
            print(f"  note: {note}")

        graph = build_knn_graph(prepared.frame, prepared.feature_columns, k=args.k)
        result = run_detection(
            graph.features,
            graph.edge_index,
            architectures,
            contamination=args.contamination,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
            threshold_strategy=args.threshold_strategy,
            progress_callback=_print_progress,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    explanations = top_feature_contributions(result.feature_errors, prepared.feature_columns, top_k=args.top_features)
    output = attach_results(loaded.frame, result.scores, result.is_outlier, explanations)
    if args.outliers_only:
        output = output[output["is_outlier"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(writer(output))

    if args.db:
        engine = get_engine(args.db)
        run_id = save_run(
            engine,
            loaded.source_name,
            loaded.frame,
            result.scores,
            result.is_outlier,
            architectures,
            result.used_backend,
            result.threshold,
            feature_count=len(prepared.feature_columns),
            params={
                "k": args.k,
                "contamination": args.contamination,
                "epochs": args.epochs,
                "hidden_dim": args.hidden_dim,
                "seed": args.seed,
                "threshold_strategy": args.threshold_strategy,
            },
            persist_mode=PERSIST_ALL if args.persist_all else PERSIST_OUTLIERS,
        )
        print(f"persisted run #{run_id} to {args.db}")

    flagged = int(result.is_outlier.sum())
    print(f"backend: {result.used_backend}")
    print(f"scored {len(result.scores)} rows; flagged {flagged} outliers (threshold {result.threshold:.4f}, strategy {result.threshold_strategy})")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
