# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Summarize Megatron FSDP/MoE forward timings from an Nsight Systems report.

The script accepts either a Nsight Systems ``.nsys-rep`` file or a SQLite export.
For ``.nsys-rep`` inputs it runs ``nsys export`` first, so the Nsight Systems CLI
must be available on PATH.

Timing attribution is based on CUDA launch correlation IDs when the SQLite export
contains CUDA runtime rows. In that mode, a kernel belongs to an NVTX range when
the CUDA runtime launch carrying the same correlation ID happened inside the
range. If runtime rows are unavailable, the script falls back to GPU timestamp
overlap with the NVTX interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NS_PER_MS = 1_000_000.0

LAYER_ATTENTION_RE = re.compile(r"layer=(?P<layer>\d+)\.attention_compute")
LAYER_MLP_RE = re.compile(r"layer=(?P<layer>\d+)\.(?P<kind>moe|mlp)_compute")
FSDP_FORWARD_RE = re.compile(r"CustomFSDP\.forward")
FSDP_OP_ID_RE = re.compile(r"op_id\s*=\s*(?P<op_id>[^,\s]+)")
PREFETCH_LAYER_RE = re.compile(r"prefetch\.layer=(?P<layer>\d+)")


@dataclass(frozen=True)
class NvtxRange:
    """A completed NVTX range from the Nsight SQLite export."""

    row_id: int
    start: int
    end: int
    text: str
    global_tid: int | None = None

    @property
    def duration_ns(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RuntimeEvent:
    """A CUDA runtime/driver API event that may launch a GPU kernel."""

    start: int
    end: int
    correlation_id: int
    global_tid: int | None = None


@dataclass(frozen=True)
class KernelEvent:
    """A CUDA GPU kernel event."""

    start: int
    end: int
    name: str
    correlation_id: int | None = None

    @property
    def duration_ns(self) -> int:
        return self.end - self.start


@dataclass
class KernelStats:
    """Aggregated GPU kernel time."""

    total_ns: float = 0.0
    count: float = 0.0
    names: Counter[str] = field(default_factory=Counter)

    def add_kernel(self, kernel: KernelEvent, scale: float = 1.0) -> None:
        self.total_ns += kernel.duration_ns * scale
        self.count += scale
        self.names[kernel.name] += scale

    def add_stats(self, other: "KernelStats", scale: float = 1.0) -> None:
        self.total_ns += other.total_ns * scale
        self.count += other.count * scale
        for name, value in other.names.items():
            self.names[name] += value * scale

    @property
    def total_ms(self) -> float:
        return self.total_ns / NS_PER_MS


@dataclass
class LayerOccurrence:
    """One layer execution inside one FSDP forward range."""

    forward_index: int | None
    forward_op_id: str
    layer: int
    occurrence: int
    attention: NvtxRange
    mlp: NvtxRange | None = None
    mlp_kind: str = ""
    layer_start: int = 0
    layer_end: int = 0
    layer_kernels: KernelStats = field(default_factory=KernelStats)
    attention_kernels: KernelStats = field(default_factory=KernelStats)
    mlp_kernels: KernelStats = field(default_factory=KernelStats)
    dispatch_kernels: KernelStats = field(default_factory=KernelStats)
    combine_kernels: KernelStats = field(default_factory=KernelStats)
    prefetch_kernels: KernelStats = field(default_factory=KernelStats)

    @property
    def key(self) -> tuple[int | None, int, int]:
        return (self.forward_index, self.layer, self.occurrence)


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def normalize_columns(columns: list[str]) -> dict[str, str]:
    return {column.lower(): column for column in columns}


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = normalize_columns(columns)
    for candidate in candidates:
        column = normalized.get(candidate.lower())
        if column is not None:
            return column
    return None


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    return [str(row[1]) for row in rows]


def load_string_table(conn: sqlite3.Connection) -> dict[int, str]:
    """Load Nsight string IDs if the export contains a string table."""

    for table in table_names(conn):
        columns = table_columns(conn, table)
        id_col = find_column(columns, ["id", "stringId"])
        value_col = find_column(columns, ["value", "string", "text", "name"])
        if table.lower() in {"stringids", "strings"} and id_col and value_col:
            query = f"SELECT {quote_identifier(id_col)}, {quote_identifier(value_col)} FROM {quote_identifier(table)}"
            return {int(row[0]): str(row[1]) for row in conn.execute(query) if row[1] is not None}
    return {}


def resolve_text(row: sqlite3.Row, columns: list[str], strings: dict[int, str]) -> str:
    """Resolve an inline or string-ID-backed text/name field from a SQLite row."""

    direct_candidates = [
        "text",
        "message",
        "name",
        "label",
        "demangledName",
        "shortName",
        "mangledName",
        "textId",
        "nameId",
    ]
    for candidate in direct_candidates:
        column = find_column(columns, [candidate])
        if column is None:
            continue
        value = row[column]
        if value is None:
            continue
        if isinstance(value, str):
            if value:
                return value
        else:
            try:
                resolved = strings.get(int(value))
            except (TypeError, ValueError):
                resolved = None
            if resolved:
                return resolved
    return ""


def optional_int(row: sqlite3.Row, columns: list[str], candidates: list[str]) -> int | None:
    column = find_column(columns, candidates)
    if column is None or row[column] is None:
        return None
    return int(row[column])


def required_int(row: sqlite3.Row, columns: list[str], candidates: list[str]) -> int:
    value = optional_int(row, columns, candidates)
    if value is None:
        raise ValueError(f"missing required column from candidates: {candidates}")
    return value


def select_all_with_rowid(conn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    quoted = quote_identifier(table)
    try:
        return conn.execute(f"SELECT rowid AS __rowid__, * FROM {quoted}").fetchall()
    except sqlite3.OperationalError:
        rows = conn.execute(f"SELECT * FROM {quoted}").fetchall()
        for index, row in enumerate(rows):
            row_dict = dict(row)
            row_dict["__rowid__"] = index
        return rows


def load_nvtx_ranges(conn: sqlite3.Connection, strings: dict[int, str]) -> list[NvtxRange]:
    ranges: list[NvtxRange] = []
    for table in table_names(conn):
        if "NVTX" not in table.upper():
            continue
        columns = table_columns(conn, table)
        if find_column(columns, ["start"]) is None or find_column(columns, ["end"]) is None:
            continue
        for row in select_all_with_rowid(conn, table):
            start = optional_int(row, columns, ["start"])
            end = optional_int(row, columns, ["end"])
            if start is None or end is None or end <= start:
                continue
            text = resolve_text(row, columns, strings)
            if not text:
                continue
            row_id = int(row["__rowid__"]) if "__rowid__" in row.keys() else len(ranges)
            ranges.append(
                NvtxRange(
                    row_id=row_id,
                    start=start,
                    end=end,
                    text=text,
                    global_tid=optional_int(row, columns, ["globalTid", "globalThreadId", "tid", "threadId"]),
                )
            )
    ranges.sort(key=lambda item: (item.start, item.end))
    return ranges


def load_runtime_events(conn: sqlite3.Connection) -> list[RuntimeEvent]:
    events: list[RuntimeEvent] = []
    for table in table_names(conn):
        upper = table.upper()
        if "RUNTIME" not in upper and "CUDA_API" not in upper and "DRIVER" not in upper:
            continue
        columns = table_columns(conn, table)
        if (
            find_column(columns, ["start"]) is None
            or find_column(columns, ["end"]) is None
            or find_column(columns, ["correlationId", "correlation_id"]) is None
        ):
            continue
        for row in select_all_with_rowid(conn, table):
            start = optional_int(row, columns, ["start"])
            end = optional_int(row, columns, ["end"])
            correlation_id = optional_int(row, columns, ["correlationId", "correlation_id"])
            if start is None or end is None or correlation_id is None:
                continue
            events.append(
                RuntimeEvent(
                    start=start,
                    end=end,
                    correlation_id=correlation_id,
                    global_tid=optional_int(row, columns, ["globalTid", "globalThreadId", "tid", "threadId"]),
                )
            )
    events.sort(key=lambda item: item.start)
    return events


def load_kernel_events(conn: sqlite3.Connection, strings: dict[int, str]) -> list[KernelEvent]:
    kernels: list[KernelEvent] = []
    for table in table_names(conn):
        upper = table.upper()
        if "KERNEL" not in upper or "ENUM" in upper:
            continue
        columns = table_columns(conn, table)
        if find_column(columns, ["start"]) is None or find_column(columns, ["end"]) is None:
            continue
        for row in select_all_with_rowid(conn, table):
            start = optional_int(row, columns, ["start"])
            end = optional_int(row, columns, ["end"])
            if start is None or end is None or end <= start:
                continue
            name = resolve_text(row, columns, strings) or "<unknown kernel>"
            kernels.append(
                KernelEvent(
                    start=start,
                    end=end,
                    name=name,
                    correlation_id=optional_int(row, columns, ["correlationId", "correlation_id"]),
                )
            )
    kernels.sort(key=lambda item: item.start)
    return kernels


class KernelIndexer:
    """Find kernels launched in a time window or NVTX range."""

    def __init__(self, runtimes: list[RuntimeEvent], kernels: list[KernelEvent], attribution: str) -> None:
        self.runtimes = runtimes
        self.kernels = kernels
        self.attribution = attribution
        self.runtime_starts = [runtime.start for runtime in runtimes]
        self.kernel_starts = [kernel.start for kernel in kernels]
        self.kernels_by_correlation: dict[int, list[KernelEvent]] = defaultdict(list)
        for kernel in kernels:
            if kernel.correlation_id is not None:
                self.kernels_by_correlation[kernel.correlation_id].append(kernel)

    def stats_for_range(self, nvtx_range: NvtxRange) -> KernelStats:
        return self.stats_for_window(nvtx_range.start, nvtx_range.end, nvtx_range.global_tid)

    def stats_for_window(self, start: int, end: int, global_tid: int | None = None) -> KernelStats:
        if self.attribution == "launch" and self.runtimes and self.kernels_by_correlation:
            return self._stats_by_launch(start, end, global_tid)
        return self._stats_by_gpu_overlap(start, end)

    def _stats_by_launch(self, start: int, end: int, global_tid: int | None) -> KernelStats:
        stats = KernelStats()
        seen_correlation_ids: set[int] = set()
        left = bisect_left(self.runtime_starts, start)
        index = left
        while index < len(self.runtimes) and self.runtimes[index].start < end:
            runtime = self.runtimes[index]
            index += 1
            if global_tid is not None and runtime.global_tid is not None and runtime.global_tid != global_tid:
                continue
            if runtime.correlation_id in seen_correlation_ids:
                continue
            seen_correlation_ids.add(runtime.correlation_id)
            for kernel in self.kernels_by_correlation.get(runtime.correlation_id, []):
                stats.add_kernel(kernel)
        return stats

    def _stats_by_gpu_overlap(self, start: int, end: int) -> KernelStats:
        stats = KernelStats()
        index = bisect_left(self.kernel_starts, start)
        while index > 0 and self.kernels[index - 1].end > start:
            index -= 1
        while index < len(self.kernels) and self.kernels[index].start < end:
            kernel = self.kernels[index]
            index += 1
            if kernel.end > start:
                stats.add_kernel(kernel)
        return stats


def enclosing_range(ranges: list[NvtxRange], child: NvtxRange) -> NvtxRange | None:
    candidates = [item for item in ranges if item.start <= child.start and item.end >= child.end]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item.duration_ns)


def parse_forward_op_id(text: str, fallback: int) -> str:
    match = FSDP_OP_ID_RE.search(text)
    if match:
        return match.group("op_id")
    return str(fallback)


def build_layer_occurrences(nvtx_ranges: list[NvtxRange], kernels: KernelIndexer) -> list[LayerOccurrence]:
    forward_ranges = [item for item in nvtx_ranges if FSDP_FORWARD_RE.search(item.text)]
    forward_ranges.sort(key=lambda item: item.start)
    forward_index_by_row = {item.row_id: index for index, item in enumerate(forward_ranges)}

    attention_ranges: list[tuple[int, NvtxRange]] = []
    mlp_ranges_by_forward_layer: dict[tuple[int | None, int], list[tuple[str, NvtxRange]]] = defaultdict(list)

    for item in nvtx_ranges:
        attention_match = LAYER_ATTENTION_RE.search(item.text)
        if attention_match:
            attention_ranges.append((int(attention_match.group("layer")), item))
            continue
        mlp_match = LAYER_MLP_RE.search(item.text)
        if mlp_match:
            forward = enclosing_range(forward_ranges, item)
            forward_index = forward_index_by_row.get(forward.row_id) if forward is not None else None
            key = (forward_index, int(mlp_match.group("layer")))
            mlp_ranges_by_forward_layer[key].append((mlp_match.group("kind"), item))

    attention_ranges.sort(key=lambda pair: pair[1].start)
    for ranges in mlp_ranges_by_forward_layer.values():
        ranges.sort(key=lambda pair: pair[1].start)

    occurrences: list[LayerOccurrence] = []
    seen_by_forward_layer: Counter[tuple[int | None, int]] = Counter()
    for layer, attention in attention_ranges:
        forward = enclosing_range(forward_ranges, attention)
        forward_index = forward_index_by_row.get(forward.row_id) if forward is not None else None
        forward_op_id = parse_forward_op_id(forward.text, forward_index) if forward is not None else ""
        occurrence = seen_by_forward_layer[(forward_index, layer)]
        seen_by_forward_layer[(forward_index, layer)] += 1

        mlp_kind = ""
        mlp_range: NvtxRange | None = None
        for candidate_kind, candidate_range in mlp_ranges_by_forward_layer.get((forward_index, layer), []):
            if candidate_range.start >= attention.start:
                mlp_kind = candidate_kind
                mlp_range = candidate_range
                break

        end = mlp_range.end if mlp_range is not None else attention.end
        occurrences.append(
            LayerOccurrence(
                forward_index=forward_index,
                forward_op_id=forward_op_id,
                layer=layer,
                occurrence=occurrence,
                attention=attention,
                mlp=mlp_range,
                mlp_kind=mlp_kind,
                layer_start=attention.start,
                layer_end=end,
            )
        )

    occurrences.sort(key=lambda item: (item.forward_index if item.forward_index is not None else -1, item.layer_start))
    for index, occurrence in enumerate(occurrences):
        next_occurrence = next(
            (
                candidate
                for candidate in occurrences[index + 1 :]
                if (
                    candidate.forward_index == occurrence.forward_index
                    and candidate.layer_start > occurrence.layer_start
                )
            ),
            None,
        )
        if next_occurrence is not None:
            if occurrence.mlp is None:
                occurrence.layer_end = min(next_occurrence.layer_start, occurrence.layer_end)
            else:
                occurrence.layer_end = next_occurrence.layer_start

        occurrence.layer_kernels = kernels.stats_for_window(
            occurrence.layer_start, occurrence.layer_end, occurrence.attention.global_tid
        )
        occurrence.attention_kernels = kernels.stats_for_range(occurrence.attention)
        if occurrence.mlp is not None:
            occurrence.mlp_kernels = kernels.stats_for_range(occurrence.mlp)

    return occurrences


def assign_moe_comm(
    occurrences: list[LayerOccurrence], nvtx_ranges: list[NvtxRange], kernels: KernelIndexer
) -> None:
    moe_occurrences = [item for item in occurrences if item.mlp is not None and item.mlp_kind == "moe"]
    for nvtx_range in nvtx_ranges:
        is_dispatch = bool(re.search(r"^moe\..*\.dispatch\.", nvtx_range.text))
        is_combine = bool(re.search(r"^moe\..*\.combine\.", nvtx_range.text))
        if not is_dispatch and not is_combine:
            continue
        owner = next(
            (
                item
                for item in moe_occurrences
                if item.mlp is not None and item.mlp.start <= nvtx_range.start and item.mlp.end >= nvtx_range.end
            ),
            None,
        )
        if owner is None:
            continue
        if is_dispatch:
            owner.dispatch_kernels.add_stats(kernels.stats_for_range(nvtx_range))
        if is_combine:
            owner.combine_kernels.add_stats(kernels.stats_for_range(nvtx_range))


def is_prefetch_range(text: str, direction: str) -> bool:
    if "param_all_gather" not in text or "prefetch.layer=" not in text:
        return False
    if direction != "all" and f"direction={direction}" not in text:
        return False
    return True


def assign_prefetch(
    occurrences: list[LayerOccurrence],
    nvtx_ranges: list[NvtxRange],
    kernels: KernelIndexer,
    direction: str,
    use_group_ranges: bool,
) -> int:
    """Assign FSDP parameter prefetch kernels to the target layer in the NVTX label."""

    by_layer: dict[int, list[LayerOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        by_layer[occurrence.layer].append(occurrence)

    forward_ranges = [item for item in nvtx_ranges if FSDP_FORWARD_RE.search(item.text)]
    forward_ranges.sort(key=lambda item: item.start)
    forward_index_by_row = {item.row_id: index for index, item in enumerate(forward_ranges)}

    prefetch_ranges = [item for item in nvtx_ranges if is_prefetch_range(item.text, direction)]
    leaf_ranges = [
        item
        for item in prefetch_ranges
        if re.match(r"^(outer_)?fsdp\.param_all_gather\.", item.text) and "_group." not in item.text
    ]
    group_ranges = [
        item
        for item in prefetch_ranges
        if re.match(r"^(outer_)?fsdp\.param_all_gather_group\.", item.text)
    ]

    stats_cache: dict[int, KernelStats] = {}

    def range_stats(nvtx_range: NvtxRange) -> KernelStats:
        if nvtx_range.row_id not in stats_cache:
            stats_cache[nvtx_range.row_id] = kernels.stats_for_range(nvtx_range)
        return stats_cache[nvtx_range.row_id]

    if use_group_ranges:
        selected_ranges = group_ranges
    else:
        leaf_total_ns = sum(range_stats(item).total_ns for item in leaf_ranges)
        selected_ranges = leaf_ranges if leaf_total_ns > 0 else group_ranges

    multi_layer_groups = 0
    for nvtx_range in selected_ranges:
        target_layers = sorted({int(match) for match in PREFETCH_LAYER_RE.findall(nvtx_range.text)})
        if not target_layers:
            continue
        if len(target_layers) > 1:
            multi_layer_groups += 1
        stats = range_stats(nvtx_range)
        scale = 1.0 / len(target_layers)
        for layer in target_layers:
            owners = sorted(by_layer.get(layer, []), key=lambda item: item.layer_start)
            if not owners:
                continue
            prefetch_forward = enclosing_range(forward_ranges, nvtx_range)
            prefetch_forward_index = (
                forward_index_by_row.get(prefetch_forward.row_id) if prefetch_forward is not None else None
            )
            same_forward_owners = [
                candidate for candidate in owners if candidate.forward_index == prefetch_forward_index
            ]
            if same_forward_owners:
                owners = same_forward_owners
            # FSDP prefetch for a target layer usually happens before that layer starts.
            owner = next(
                (candidate for candidate in owners if nvtx_range.start <= candidate.layer_start),
                owners[-1],
            )
            owner.prefetch_kernels.add_stats(stats, scale=scale)
    return multi_layer_groups


def format_ms(value: float) -> str:
    return f"{value:.3f}"


def percentile(values: list[float], pct: float) -> float:
    """Return an interpolated percentile for a non-empty list."""

    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def layer_rows(occurrences: list[LayerOccurrence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for occurrence in occurrences:
        rows.append(
            {
                "forward_index": "" if occurrence.forward_index is None else occurrence.forward_index,
                "forward_op_id": occurrence.forward_op_id,
                "layer": occurrence.layer,
                "occurrence": occurrence.occurrence,
                "mlp_kind": occurrence.mlp_kind,
                "layer_wall_ms": occurrence.layer_end - occurrence.layer_start,
                "layer_kernel_ms": occurrence.layer_kernels.total_ms,
                "layer_kernel_count": occurrence.layer_kernels.count,
                "attention_range_ms": occurrence.attention.duration_ns / NS_PER_MS,
                "attention_kernel_ms": occurrence.attention_kernels.total_ms,
                "attention_kernel_count": occurrence.attention_kernels.count,
                "mlp_or_moe_range_ms": occurrence.mlp.duration_ns / NS_PER_MS if occurrence.mlp is not None else 0.0,
                "mlp_or_moe_kernel_ms": occurrence.mlp_kernels.total_ms,
                "mlp_or_moe_kernel_count": occurrence.mlp_kernels.count,
                "moe_dispatch_kernel_ms": occurrence.dispatch_kernels.total_ms,
                "moe_dispatch_kernel_count": occurrence.dispatch_kernels.count,
                "moe_combine_kernel_ms": occurrence.combine_kernels.total_ms,
                "moe_combine_kernel_count": occurrence.combine_kernels.count,
                "param_prefetch_kernel_ms": occurrence.prefetch_kernels.total_ms,
                "param_prefetch_kernel_count": occurrence.prefetch_kernels.count,
                "top_prefetch_kernels": "; ".join(
                    f"{name} ({count:g})" for name, count in occurrence.prefetch_kernels.names.most_common(3)
                ),
                "top_dispatch_kernels": "; ".join(
                    f"{name} ({count:g})" for name, count in occurrence.dispatch_kernels.names.most_common(3)
                ),
                "top_combine_kernels": "; ".join(
                    f"{name} ({count:g})" for name, count in occurrence.combine_kernels.names.most_common(3)
                ),
            }
        )
        rows[-1]["layer_wall_ms"] = rows[-1]["layer_wall_ms"] / NS_PER_MS
    return rows


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["layer"])].append(row)

    metric_columns = [
        "layer_wall_ms",
        "layer_kernel_ms",
        "attention_kernel_ms",
        "mlp_or_moe_kernel_ms",
        "moe_dispatch_kernel_ms",
        "moe_combine_kernel_ms",
        "param_prefetch_kernel_ms",
    ]
    summaries: list[dict[str, Any]] = []
    for layer, layer_group in sorted(grouped.items()):
        summary: dict[str, Any] = {"layer": layer, "occurrences": len(layer_group)}
        for column in metric_columns:
            values = [float(row[column]) for row in layer_group]
            summary[f"avg_{column}"] = sum(values) / len(values)
            summary[f"median_{column}"] = percentile(values, 0.50)
            summary[f"p95_{column}"] = percentile(values, 0.95)
            summary[f"max_{column}"] = max(values)
            summary[f"sum_{column}"] = sum(values)
        summary["mlp_kind"] = ",".join(sorted({str(row["mlp_kind"]) for row in layer_group if row["mlp_kind"]}))
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], limit: int | None, statistic: str) -> None:
    selected = rows[:limit] if limit is not None else rows
    prefix = "median" if statistic == "median" else statistic
    columns = [
        ("layer", "layer"),
        ("occ", "occurrences"),
        ("cpu_wall", f"{prefix}_layer_wall_ms"),
        ("layer_gpu", f"{prefix}_layer_kernel_ms"),
        ("attn_gpu", f"{prefix}_attention_kernel_ms"),
        ("mlp_gpu", f"{prefix}_mlp_or_moe_kernel_ms"),
        ("dispatch", f"{prefix}_moe_dispatch_kernel_ms"),
        ("combine", f"{prefix}_moe_combine_kernel_ms"),
        ("prefetch", f"{prefix}_param_prefetch_kernel_ms"),
    ]
    print(f"Summary {statistic} values in ms:")
    print("  cpu_wall is the CPU NVTX interval; *_gpu columns sum attributed GPU kernel durations.")
    print("  " + " ".join(title.rjust(10) for title, _ in columns))
    for row in selected:
        rendered = []
        for title, key in columns:
            value = row[key]
            if isinstance(value, float):
                rendered.append(format_ms(value).rjust(10))
            else:
                rendered.append(str(value).rjust(10))
        print("  " + " ".join(rendered))


def print_outlier_notes(rows: list[dict[str, Any]], top_n: int = 5) -> None:
    """Print rows where one occurrence is much larger than its layer median."""

    outliers: list[tuple[float, dict[str, Any], str, float, float]] = []
    checks = [
        ("mlp_or_moe_kernel_ms", "mlp_gpu"),
        ("moe_combine_kernel_ms", "combine"),
        ("layer_kernel_ms", "layer_gpu"),
    ]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["layer"])].append(row)

    for layer_rows_for_layer in grouped.values():
        for column, label in checks:
            median = percentile([float(row[column]) for row in layer_rows_for_layer], 0.50)
            if median <= 0.0:
                continue
            for row in layer_rows_for_layer:
                value = float(row[column])
                if value >= max(50.0, median * 5.0):
                    outliers.append((value / median, row, label, value, median))

    if not outliers:
        return

    print("\nLargest long-tail occurrences:")
    for ratio, row, label, value, median in sorted(outliers, reverse=True)[:top_n]:
        print(
            "  "
            f"forward={row['forward_index']} layer={row['layer']} {label}={value:.3f} ms "
            f"(median {median:.3f} ms, {ratio:.1f}x)"
        )


def is_sqlite_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def export_nsys_report(
    report_path: Path, keep_sqlite: Path | None
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError(
            "Input is not a SQLite export and `nsys` was not found on PATH. "
            "Run this on a machine with Nsight Systems CLI, or export first with "
            "`nsys export --type sqlite --output report.sqlite report.nsys-rep`."
        )

    if keep_sqlite is not None:
        sqlite_path = keep_sqlite
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="nsys_export_")
        sqlite_path = Path(temp_dir.name) / f"{report_path.stem}.sqlite"

    command = [
        nsys,
        "export",
        "--type",
        "sqlite",
        "--force-overwrite",
        "true",
        "--output",
        str(sqlite_path),
        str(report_path),
    ]
    subprocess.run(command, check=True)
    if sqlite_path.exists():
        return sqlite_path, temp_dir

    alternate = sqlite_path.with_suffix(sqlite_path.suffix + ".sqlite")
    if alternate.exists():
        return alternate, temp_dir
    raise RuntimeError(f"`nsys export` completed, but no SQLite file was found at {sqlite_path}")


def prepare_sqlite(input_path: Path, keep_sqlite: Path | None) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if is_sqlite_file(input_path):
        return input_path, None
    return export_nsys_report(input_path, keep_sqlite)


def open_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Input .nsys-rep file or Nsight SQLite export.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Prefix for generated .layers.csv, .summary.csv, and .json files. "
            "Defaults to ./<input-stem>.layer_profile."
        ),
    )
    parser.add_argument("--keep-sqlite", type=Path, default=None, help="Keep the exported SQLite file at this path.")
    parser.add_argument(
        "--attribution",
        choices=["launch", "overlap"],
        default="launch",
        help="Kernel attribution mode. `launch` uses CUDA correlation IDs; `overlap` uses GPU timestamp overlap.",
    )
    parser.add_argument(
        "--prefetch-direction",
        choices=["fwd", "bwd", "all"],
        default="fwd",
        help="Which FSDP param prefetch direction to include.",
    )
    parser.add_argument(
        "--use-prefetch-group-ranges",
        action="store_true",
        help=(
            "Use fsdp.param_all_gather_group ranges for prefetch attribution. "
            "By default, leaf ranges are used if present."
        ),
    )
    parser.add_argument("--no-output-files", action="store_true", help="Only print the summary table.")
    parser.add_argument("--print-limit", type=int, default=None, help="Limit the number of printed summary rows.")
    parser.add_argument(
        "--print-stat",
        choices=["avg", "median", "p95", "max"],
        default="avg",
        help="Statistic to print in the terminal summary. CSV/JSON outputs include all statistics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_prefix = args.output_prefix or Path.cwd() / f"{input_path.stem}.layer_profile"

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        sqlite_path, temp_dir = prepare_sqlite(input_path, args.keep_sqlite)
        with open_sqlite(sqlite_path) as conn:
            strings = load_string_table(conn)
            nvtx_ranges = load_nvtx_ranges(conn, strings)
            runtimes = load_runtime_events(conn)
            kernels = load_kernel_events(conn, strings)

        if not nvtx_ranges:
            raise RuntimeError("No completed NVTX ranges were found in the SQLite export.")
        if not kernels:
            raise RuntimeError("No CUDA kernel events were found in the SQLite export.")

        kernel_indexer = KernelIndexer(runtimes, kernels, args.attribution)
        occurrences = build_layer_occurrences(nvtx_ranges, kernel_indexer)
        if not occurrences:
            raise RuntimeError("No `layer=N.attention_compute` NVTX ranges were found.")

        assign_moe_comm(occurrences, nvtx_ranges, kernel_indexer)
        multi_layer_groups = assign_prefetch(
            occurrences,
            nvtx_ranges,
            kernel_indexer,
            args.prefetch_direction,
            args.use_prefetch_group_ranges,
        )

        detail = layer_rows(occurrences)
        summary = summary_rows(detail)
        print_summary(summary, args.print_limit, args.print_stat)
        print_outlier_notes(detail)

        if multi_layer_groups:
            print(
                f"\nNote: split {multi_layer_groups} prefetch range(s) that referenced multiple target layers "
                "equally across those layers."
            )
        if args.attribution == "launch" and not runtimes:
            print("\nNote: no CUDA runtime table was found; fell back to GPU timestamp overlap attribution.")

        if not args.no_output_files:
            detail_csv = output_prefix.with_suffix(".layers.csv")
            summary_csv = output_prefix.with_suffix(".summary.csv")
            json_path = output_prefix.with_suffix(".json")
            write_csv(detail_csv, detail)
            write_csv(summary_csv, summary)
            with json_path.open("w") as output:
                json.dump({"layers": detail, "summary": summary}, output, indent=2)
            print(f"\nWrote {detail_csv}")
            print(f"Wrote {summary_csv}")
            print(f"Wrote {json_path}")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
