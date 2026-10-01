#!/usr/bin/env python3
"""Compare the merge-check runs: exact-path output and memory, fixes off vs on.

Exits non-zero if any generated text differs, so a chained script can stop.
"""
import json
import os
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "results", "merge_check")


def rows(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d.get("key")] = d
    return out


def main():
    ok = True
    for tag in ("qwen16k", "granite8k"):
        off, on = rows(tag + "_off.jsonl"), rows(tag + "_on.jsonl")
        keys = sorted(set(off) & set(on))
        diff = [k for k in keys if off[k].get("text") != on[k].get("text")]
        print(f"{tag}: {len(keys)} shared items, {len(diff)} with different text")
        for k in diff[:5]:
            print(f"   {k}\n     off: {off[k].get('text', '')[:90]!r}\n"
                  f"     on:  {on[k].get('text', '')[:90]!r}")
        ok &= bool(keys) and not diff
    for tag in ("granite_ladder16k", "qwen_ladder64k"):
        for side in ("off", "on"):
            for d in rows(f"{tag}_{side}.jsonl").values():
                print(f"{tag} {side}: status {d.get('status')} peak "
                      f"{d.get('peak_gb', 0):.6f} reserved "
                      f"{d.get('peak_reserved_gb', 0):.6f} GB, wall "
                      f"{d.get('wall_s', 0):.1f}s")
    print("MERGE CHECK", "PASS (identical text)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
