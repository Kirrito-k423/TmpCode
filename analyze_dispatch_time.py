import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PHASE_NAMES = [
    # "AlltoAllDispatch init / SendCnt / get_sqs",
    # "ExpIdsCopy / WaitDispatch / SimtPrepareMapping",
    # "SendtoExpertNew / CalRecvAndSetFlag / PollingAndSend",
    # "Null / SetExpertTokenNums / Null",
    # "LocalWindowCopy",
]

DEFAULT_FILE_PREFIX = "dispatch_clock"
DEFAULT_LATENCY_FILE = "dispatch_latency.csv"


def find_input_files(input_dir: Path, file_prefix: str) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")

    patterns = [
        f"{file_prefix}_rank_*.csv",
        f"{file_prefix}_*_rank_*.csv",
    ]

    def raw_clock_files(paths) -> List[Path]:
        return sorted(path for path in paths if not path.name.endswith("_summary.csv"))

    for pattern in patterns:
        paths = raw_clock_files(input_dir.glob(pattern))
        if paths:
            return paths

    # The normal layout puts clock CSV files in <run_dir>/dispatch_clock/.
    for pattern in patterns:
        paths = raw_clock_files(input_dir.rglob(pattern))
        if paths:
            return paths
    return []


def time_col_index(name: str) -> int:
    # Expected format: time_0_cycles
    return int(name.split("_")[1])


def load_clock_csv(path: Path) -> Tuple[int, np.ndarray, List[int], List[str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")

        time_cols = [
            name for name in reader.fieldnames
            if name.startswith("time_") and name.endswith("_cycles")
        ]
        time_cols = sorted(time_cols, key=time_col_index)
        if not time_cols:
            raise ValueError(f"{path} has no time_*_cycles columns")

        rows = []
        for row in reader:
            rank = int(row["rank"])
            aiv_id = int(row["aiv_id"])
            values = [int(row[col]) for col in time_cols]
            rows.append((rank, aiv_id, values))

    if not rows:
        raise ValueError(f"{path} has no data rows")

    ranks = {row[0] for row in rows}
    if len(ranks) != 1:
        raise ValueError(f"{path} contains multiple ranks: {sorted(ranks)}")

    rows.sort(key=lambda item: item[1])
    rank = rows[0][0]
    aiv_ids = [row[1] for row in rows]
    data = np.array([row[2] for row in rows], dtype=np.int64)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (data[i,j] < 0 or data[i,j] > 500000):
                data[i,j] = 0
    return rank, data, aiv_ids, time_cols


def make_phase_names(raw_names: List[str], valid_slots: int, custom_names: List[str]) -> List[str]:
    names: List[str] = []
    for i in range(valid_slots):
        if i < len(custom_names):
            names.append(custom_names[i])
        elif i < len(DEFAULT_PHASE_NAMES):
            names.append(DEFAULT_PHASE_NAMES[i])
        else:
            names.append(raw_names[i].replace("_cycles", ""))
    return names


def sanitize_cycles(data: np.ndarray, source: Path) -> np.ndarray:
    negative_count = int(np.sum(data < 0))
    if negative_count > 0:
        print(f"[WARN] {source}: clipped {negative_count} negative cycle values to 0")
        data = data.copy()
        data[data < 0] = 0
    return data


def write_summary_csv(
    output_path: Path,
    rank: int,
    aiv_ids: List[int],
    data_us: np.ndarray,
    phase_names: List[str],
) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "aiv_id", *[f"{name}_us" for name in phase_names], "total_us"])
        for aiv_id, row in zip(aiv_ids, data_us):
            writer.writerow([rank, aiv_id, *[f"{value:.6f}" for value in row], f"{np.sum(row):.6f}"])


def print_stats(rank: int, aiv_ids: List[int], data_us: np.ndarray, phase_names: List[str]) -> None:
    print(f"\nrank {rank}")
    print("phase, mean_us, p50_us, p95_us, max_us")
    for idx, name in enumerate(phase_names):
        col = data_us[:, idx]
        print(
            f"{name}, "
            f"{np.mean(col):.3f}, "
            f"{np.percentile(col, 50):.3f}, "
            f"{np.percentile(col, 95):.3f}, "
            f"{np.max(col):.3f}"
        )
    total = np.sum(data_us, axis=1)
    slowest_idx = int(np.argmax(total))
    slowest_aiv = aiv_ids[slowest_idx]
    print(
        f"total, mean={np.mean(total):.3f} us, "
        f"p95={np.percentile(total, 95):.3f} us, "
        f"max={np.max(total):.3f} us at aiv {slowest_aiv}"
    )


def plot_rank(
    output_path: Path,
    rank: int,
    aiv_ids: List[int],
    data_us: np.ndarray,
    phase_names: List[str],
    title: str,
) -> None:
    fig_height = max(7.0, len(aiv_ids) * 0.22)
    fig, ax = plt.subplots(figsize=(15, fig_height))

    y_pos = np.arange(len(aiv_ids))
    left = np.zeros(len(aiv_ids), dtype=np.float64)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(phase_names), 2)))

    for idx, phase in enumerate(phase_names):
        ax.barh(
            y_pos,
            data_us[:, idx],
            left=left,
            height=0.78,
            color=colors[idx],
            label=phase,
        )
        left += data_us[:, idx]

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"aiv{aiv_id}" for aiv_id in aiv_ids])
    ax.invert_yaxis()
    ax.set_xlabel("time (us)")
    ax.set_title(title or f"dispatch clock rank {rank}")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def find_latency_csv(input_dir: Path, file_name: str) -> Path:
    direct_path = input_dir / file_name
    if direct_path.is_file():
        return direct_path

    legacy_matches = sorted(input_dir.glob("dispatch_latency_*.csv"))
    if not legacy_matches:
        legacy_matches = sorted(input_dir.rglob(file_name))
    if not legacy_matches:
        legacy_matches = sorted(input_dir.rglob("dispatch_latency_*.csv"))
    if not legacy_matches:
        raise FileNotFoundError(f"No latency CSV found under {input_dir}")
    if len(legacy_matches) > 1:
        selected = max(legacy_matches, key=lambda path: path.stat().st_mtime)
        print(f"[WARN] Found {len(legacy_matches)} latency CSV files; using newest: {selected}")
        return selected
    return legacy_matches[0]


def load_latency_rows(csv_path: Path) -> List[dict]:
    required_columns = {"rank", "iteration", "elapsed_us"}
    rows: List[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        missing = required_columns - fieldnames
        if missing:
            raise ValueError(
                f"{csv_path} is missing columns: {', '.join(sorted(missing))}"
            )
        for item in reader:
            rows.append(
                {
                    "rank": int(item["rank"]),
                    "iteration": int(item["iteration"]),
                    "elapsed_us": float(item["elapsed_us"]),
                    "is_warmup": bool(int(item.get("is_warmup", "0") or "0")),
                    "in_average": bool(int(item.get("in_average", "1") or "1")),
                }
            )
    if not rows:
        raise ValueError(f"{csv_path} contains no latency rows")
    return rows


def plot_latency(rows: List[dict], csv_path: Path, output_path: Path) -> Tuple[Dict[int, float], int]:
    rows_by_rank = defaultdict(list)
    for row in rows:
        rows_by_rank[row["rank"]].append(row)

    rank_means: Dict[int, float] = {}
    measured_sample_count = 0
    for rank, rank_rows in rows_by_rank.items():
        measured_values = [row["elapsed_us"] for row in rank_rows if row["in_average"]]
        if not measured_values:
            raise ValueError(f"Rank {rank} has no samples participating in the average")
        rank_means[rank] = mean(measured_values)
        measured_sample_count += len(measured_values)

    figure, axis = plt.subplots(figsize=(12, 7))
    color_map = plt.get_cmap("tab10")
    max_iteration = max(row["iteration"] for row in rows)

    for color_index, rank in enumerate(sorted(rows_by_rank)):
        rank_rows = sorted(rows_by_rank[rank], key=lambda row: row["iteration"])
        color = color_map(color_index % 10)
        iterations = [row["iteration"] for row in rank_rows]
        latencies = [row["elapsed_us"] for row in rank_rows]
        axis.scatter(
            iterations,
            latencies,
            s=34,
            alpha=0.82,
            color=color,
            label=f"Rank {rank}",
            zorder=3,
        )
        axis.axhline(
            rank_means[rank],
            color="red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            zorder=2,
        )

        warmup_rows = [row for row in rank_rows if row["is_warmup"]]
        if warmup_rows:
            axis.scatter(
                [row["iteration"] for row in warmup_rows],
                [row["elapsed_us"] for row in warmup_rows],
                s=55,
                marker="x",
                linewidths=1.5,
                color=color,
                zorder=4,
            )

    warmup_iterations = sorted({row["iteration"] for row in rows if row["is_warmup"]})
    if warmup_iterations:
        warmup_end = max(warmup_iterations)
        axis.axvspan(
            -0.5,
            warmup_end + 0.5,
            color="gray",
            alpha=0.09,
            label="Warmup iterations",
            zorder=1,
        )
        axis.axvline(
            warmup_end + 0.5,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            zorder=2,
        )

    mean_text = "Measured mean\n" + "\n".join(
        f"Rank {rank}: {rank_means[rank]:.3f} us" for rank in sorted(rank_means)
    )
    axis.text(
        0.985,
        0.98,
        mean_text,
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.88, "edgecolor": "gray"},
        zorder=5,
    )

    if max_iteration < 50:
        axis.set_xticks(range(max_iteration + 1))
    axis.set_title(f"Dispatch latency by rank and iteration\n{csv_path.parent.name}")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Latency (us)")
    axis.set_ylim([0,200])
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    axis.legend(loc="best", frameon=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return rank_means, measured_sample_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dispatch clock and latency CSV files.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Run directory containing dispatch_clock/ or a directory containing the CSV files directly.",
    )
    parser.add_argument("--file-prefix", default=DEFAULT_FILE_PREFIX, help="Clock CSV file prefix.")
    parser.add_argument("--slots", type=int, default=11, help="Number of time columns to analyze.")
    parser.add_argument("--cycle-us", type=float, default=0.001, help="Microseconds per cycle.")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Directory for plots and summary CSV files. Default: <input_dir>/plots",
    )
    parser.add_argument(
        "--phase-names",
        nargs="*",
        default=[],
        help="Optional phase names for valid slots, for example: --phase-names send local_copy",
    )
    parser.add_argument("--title", default="", help="Optional plot title prefix.")
    parser.add_argument(
        "--latency-file",
        default=DEFAULT_LATENCY_FILE,
        help=f"Latency CSV filename (default: {DEFAULT_LATENCY_FILE}).",
    )
    parser.add_argument("--skip-clock", action="store_true", help="Skip clock CSV analysis.")
    parser.add_argument("--skip-latency", action="store_true", help="Skip latency CSV analysis.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else input_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    analyzed_anything = False

    if not args.skip_clock:
        clock_paths = find_input_files(input_dir, args.file_prefix)
        if not clock_paths:
            print(f"[INFO] No clock CSV files found in {input_dir}; skipping clock plots")
        for path in clock_paths:
            rank, cycles, aiv_ids, raw_names = load_clock_csv(path)
            valid_slots = min(args.slots, cycles.shape[1])
            cycles = sanitize_cycles(cycles[:, :valid_slots], path)
            data_us = cycles.astype(np.float64) * args.cycle_us
            phase_names = make_phase_names(raw_names, valid_slots, args.phase_names)

            summary_path = out_dir / f"{path.stem}_summary.csv"
            plot_path = out_dir / f"{path.stem}.png"

            write_summary_csv(summary_path, rank, aiv_ids, data_us, phase_names)
            print_stats(rank, aiv_ids, data_us, phase_names)
            plot_title = f"{args.title} rank {rank}".strip() if args.title else ""
            plot_rank(plot_path, rank, aiv_ids, data_us, phase_names, plot_title)

            print(f"saved summary: {summary_path}")
            print(f"saved plot: {plot_path}")
            analyzed_anything = True

    latency_means: Dict[int, float] = {}
    latency_csv = None
    latency_plot = None
    measured_sample_count = 0
    if not args.skip_latency:
        try:
            latency_csv = find_latency_csv(input_dir, args.latency_file)
        except FileNotFoundError:
            print(f"[INFO] No latency CSV found in {input_dir}; skipping latency plot")
        else:
            latency_plot = out_dir / "dispatch_latency_scatter.png"
            latency_rows = load_latency_rows(latency_csv)
            latency_means, measured_sample_count = plot_latency(
                latency_rows, latency_csv, latency_plot
            )
            analyzed_anything = True

    if not analyzed_anything:
        raise SystemExit(f"No dispatch clock or latency CSV files found in {input_dir}")

    if latency_means:
        print("\nDispatch latency measured means (warmup excluded)")
        print(f"CSV: {latency_csv}")
        print(f"Measured samples: {measured_sample_count}")
        for rank in sorted(latency_means):
            print(f"Rank {rank}: {latency_means[rank]:.6f} us")
        print(f"saved latency plot: {latency_plot}")


if __name__ == "__main__":
    main()
