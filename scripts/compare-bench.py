#!/usr/bin/env python3
"""Compare two scripts/upgrade-bench.sh runs and print a markdown table.

usage: compare-bench.py <before_dir>[,<rerun_dir>...] <after_dir>

Later before-dirs override earlier ones, so a rerun of a few models can be laid
over the full baseline. Prompt and generation rates are llama-benchy means
(t/s); with 3 runs per test, differences under ~5% are within noise. Stdlib
only."""
import json
import os
import re
import sys


def load_dir(d):
    out = {}
    for f in os.listdir(d):
        if f.endswith(".json"):
            with open(os.path.join(d, f)) as fh:
                out[f[:-5]] = json.load(fh)
    return out


def load_times(d):
    times = {}
    p = os.path.join(d, "SUMMARY.txt")
    if os.path.exists(p):
        with open(p) as fh:
            for line in fh:
                m = re.match(r"(\S+)\s+load=(\d+)s", line)
                if m:
                    times[m.group(1)] = int(m.group(2))
    return times


def rows(doc):
    return {
        (b["prompt_size"], b["concurrency"]): (b["pp_throughput"]["mean"], b["tg_throughput"]["mean"])
        for b in doc["benchmarks"]
    }


def pct(a, b):
    return f"{(b - a) / a * 100:+.0f}%" if a else "n/a"


def compare(before_dirs, after_dir):
    before, bload = {}, {}
    for d in before_dirs:
        before.update(load_dir(d))
        bload.update(load_times(d))
    after, aload = load_dir(after_dir), load_times(after_dir)

    lines = [
        "| Model | Load before → after | Test | Prompt t/s before → after | Gen t/s before → after |",
        "|---|---|---|---|---|",
    ]
    for m in sorted(before):
        if m not in after:
            lines.append(f"| {m} | not run after | | | |")
            continue
        rb, ra = rows(before[m]), rows(after[m])
        first = True
        for k in sorted(rb):
            if k not in ra:
                continue
            (pb, tb), (pa, ta) = rb[k], ra[k]
            name = m if first else ""
            ld = f"{bload.get(m, '?')}s → {aload.get(m, '?')}s" if first else ""
            lines.append(
                f"| {name} | {ld} | pp{k[0]} c{k[1]} | {pb:,.0f} → {pa:,.0f} ({pct(pb, pa)}) "
                f"| {tb:.1f} → {ta:.1f} ({pct(tb, ta)}) |"
            )
            first = False
    for m in sorted(set(after) - set(before)):
        lines.append(f"| {m} | not run before | | | |")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: compare-bench.py <before_dir>[,<rerun_dir>...] <after_dir>")
    print(compare(sys.argv[1].split(","), sys.argv[2]))
