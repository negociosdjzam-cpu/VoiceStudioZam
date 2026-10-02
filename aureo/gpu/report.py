"""Warm GPU trial report: load once, individual seed variants and saved WAVs."""


def _number(value):
    return "—" if value is None else f"{value:.3f}"


def render_report(receipt):
    lines = ["# AUREO GPU — Chatterbox Multilingual", "",
             f"Run: `{receipt['run_id']}`", "",
             "Persistent worker. The model stays loaded between variants.",
             f"Model startup/load (once): {_number(receipt['model_startup']['load_seconds'])} s.", "",
             "| Style | Seed | Passed | First PCM (s) | Total (s) | Audio (s) | RTF | RAM (MiB) | VRAM (MiB) | Reference cache | WAV |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in receipt["results"]:
        memory = row["memory"]
        cells = [row["style"], str(row["seed"]), "yes" if row["success"] else "no",
                 *[_number(row[field]) for field in ("time_to_first_audio_seconds", "total_seconds", "generated_duration_seconds", "real_time_factor")],
                 *[_number(None if memory[field] is None else memory[field] / 1024**2) for field in ("ram_peak_bytes", "vram_peak_bytes")],
                 "—" if row["reference_cache_hit"] is None else "hit" if row["reference_cache_hit"] else "miss",
                 row["wav"] or "—"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Errors and capability notes", ""]
    for row in receipt["results"]:
        notes = [f"{error['stage']}/{error['code']}: {error['message']}" for error in row["errors"]]
        if row["omitted_controls"]:
            notes.append("Unsupported preset controls: " + ", ".join(row["omitted_controls"]))
        if notes:
            safe = "; ".join(notes).replace("\n", " ").replace("\r", " ")
            lines.append(f"- {row['style']} / seed {row['seed']}: {safe}")
    lines += ["", "First PCM is measured after preload; this SDK returns buffered audio at synthesis completion.",
              "Per-variant total includes reference preparation, generation, marking and WAV save; model load is recorded once above.",
              "RAM is sampled worker RSS; VRAM is sampled PyTorch CUDA allocation. Missing values are unavailable, not zero.",
              "RTF is generation seconds / audio seconds. Fidelity is a guidance proxy, not a quality/similarity measurement.",
              "JSON includes reference cache timings, marking status, complete controls and errors. FAST/PRO/ULTRA quality assignments require listening and real GPU measurements.", ""]
    return "\n".join(lines)
