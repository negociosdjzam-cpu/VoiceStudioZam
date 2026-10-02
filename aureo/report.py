"""Versioned JSON receipts and readable comparisons; no quality scores invented."""
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
import tempfile


def _cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _number(value):
    return "—" if value is None else f"{value:.3f}"


def render_report(receipt):
    lines = [
        "# AUREO VOICE ENGINE BENCHMARK", "", f"Run: `{_cell(receipt['run_id'])}`", "",
        "Fresh process and cold engine per trial. Medians use successful trials only.", "",
        "| Engine | Style | Passed/total | Load (s) | First PCM (s) | Total (s) | Audio (s) | RTF | RAM peak (MiB) | VRAM peak (MiB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    groups = defaultdict(list)
    for result in receipt["results"]:
        groups[(result["engine_id"], result["style"])].append(result)
    for (engine, style), rows in groups.items():
        good = [row for row in rows if row["success"]]
        cells = [engine, style, f"{len(good)}/{len(rows)}"]
        for field in ("load_seconds", "time_to_first_audio_seconds", "total_seconds", "generated_duration_seconds", "real_time_factor"):
            values = [row[field] for row in good if row[field] is not None]
            cells.append(_number(statistics.median(values) if values else None))
        for field in ("ram_peak_bytes", "vram_peak_bytes"):
            values = [row["memory"][field] for row in good if row["memory"][field] is not None]
            cells.append(_number(max(values) / 1024**2 if values else None))
        lines.append("| " + " | ".join(_cell(value) for value in cells) + " |")
    lines += ["", "## Capabilities and errors", ""]
    for (engine, style), rows in groups.items():
        omitted = sorted({field for row in rows for field in row["omitted_controls"]})
        errors = sorted({f"{error['stage']}/{error['code']}: {error['message']}" for row in rows for error in row["errors"]})
        notes = []
        if omitted:
            notes.append("Unsupported preset controls omitted explicitly: " + ", ".join(omitted))
        notes.extend(errors)
        lines.append(f"- {_cell(engine)} / {_cell(style)}: " + ("; ".join(_cell(note) for note in notes) if notes else "All requested controls applied; no errors."))
    lines += [
        "", "## Interpretation", "",
        "First PCM includes engine loading; buffered SDKs return their first waveform at completion. No initial bridge advertises real streaming.",
        "Total is the engine run inside the worker; process_wall_seconds in JSON also includes interpreter startup and IPC.",
        "RAM is sampled worker RSS. VRAM is sampled PyTorch-allocated CUDA memory; unavailable values are blank, not zero.",
        "RTF is generation time / generated audio duration. Speed means speech rate, so compare equal speed/settings when judging performance.",
        "Fidelity is a model-specific control/proxy, not a measured speaker-similarity score. Unsupported controls make style intent non-equivalent across engines.",
        "FAST, PRO and ULTRA remain unassigned by this report: latency alone cannot establish cloning fidelity or speech quality.",
        "Quality/listening assessment and real GPU trials are needed before selecting final profile bindings.", "",
    ]
    return "\n".join(lines)


def validate_output_paths(output: Path, report: Path | None = None, *, overwrite=False, protected_paths=()):
    output = Path(output).resolve()
    report = output.with_suffix(".md") if report is None else Path(report).resolve()
    protected = {Path(path).resolve() for path in protected_paths}
    inside_assets = any(path.is_dir() and (output.is_relative_to(path) or report.is_relative_to(path)) for path in protected)
    if output == report or output in protected or report in protected or inside_assets:
        raise ValueError("Result files must be distinct from each other and input files")
    if not overwrite and (output.exists() or report.exists()):
        raise FileExistsError("Use explicit overwrite for previous result files")
    return output, report


def save_receipt(receipt, output: Path, report: Path | None = None, *, overwrite=False, protected_paths=()):
    output, report = validate_output_paths(output, report, overwrite=overwrite, protected_paths=protected_paths)
    content = ((output, json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"), (report, render_report(receipt)))
    staged = []
    try:
        # Stage both complete files before replacing either; each replacement
        # is atomic. The shared run_id identifies a pair after a filesystem fault.
        for destination, text in content:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
                staged.append((Path(handle.name), destination))
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, destination in staged:
            if overwrite:
                os.replace(temporary, destination)
            else:
                os.link(temporary, destination)  # atomic no-clobber publication
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return output, report
