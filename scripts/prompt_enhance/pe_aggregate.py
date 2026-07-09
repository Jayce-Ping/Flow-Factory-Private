"""Reassemble PE shard outputs ({idx, rec} per line) into a single JSONL in original order.
Verifies the total count matches the input and that no prompt is empty."""
import argparse
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_glob", required=True)
    ap.add_argument("--n_expected", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    by_idx = {}
    for path in sorted(glob.glob(args.shards_glob)):
        with open(path) as f:
            for ln in f:
                if not ln.strip():
                    continue
                o = json.loads(ln)
                by_idx[o["idx"]] = o["rec"]

    n = len(by_idx)
    if n != args.n_expected:
        raise SystemExit(f"COUNT MISMATCH: got {n}, expected {args.n_expected} (missing shards?)")

    n_changed = 0
    with open(args.output, "w") as f:
        for i in sorted(by_idx):
            rec = by_idx[i]
            if not rec.get("prompt"):
                raise SystemExit(f"empty prompt at idx {i}")
            if rec.get("prompt") != rec.get("orig_prompt"):
                n_changed += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"aggregated {n} records -> {args.output} ({n_changed} rewritten, {n - n_changed} kept-original)")


if __name__ == "__main__":
    main()
