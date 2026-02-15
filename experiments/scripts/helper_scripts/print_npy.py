#!/usr/bin/env python3
"""Print contents of .npy (or .npz) files to the console.

Usage:
    python experiments/scripts/print_npy.py file.npy [other.npy ...]

Options:
    -s, --summary   Print a short summary (shape, dtype, basic stats, histogram)
    -d, --diff      Print per-row diffs compared to the previous row
    -l, --limit N   If provided, print at most N elements from each array
"""

from pathlib import Path
import argparse
import sys
import numpy as np


def draw_histogram(arr: np.ndarray, bins: int = 40) -> str:
    """Creates a single-line ASCII histogram."""
    try:
        # Use float64 for histogram calculation to avoid overflow
        counts, _ = np.histogram(arr.astype(np.float64), bins=bins)
        
        # Normalize counts to pick characters
        # Blocks:  , ▂, ▃, ▄, ▅, ▆, ▇, █
        chars = "  ▂▃▄▅▆▇█"
        max_count = counts.max()
        if max_count == 0:
            return ""
            
        result = []
        for c in counts:
            # Map count to 0-7 index
            idx = int((c / max_count) * (len(chars) - 1))
            result.append(chars[idx])
        return "".join(result)
    except Exception:
        return ""


def print_array(path: Path, arr: np.ndarray, args: argparse.Namespace) -> None:
    print(f"--- {path} ---")
    
    if args.summary:
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        print(f"shape: {shape}, dtype: {dtype}")
        
        try:
            if np.issubdtype(arr.dtype, np.number):
                # FIX: Cast to float64 for stats to avoid float16 overflow (inf)
                stats = arr.astype(np.float64)
                
                # Basic Stats
                mn = stats.min()
                mx = stats.max()
                mean = stats.mean()
                std = stats.std()
                
                # Percentiles for better "sense" of deviation
                p25, p50, p75 = np.percentile(stats, [25, 50, 75])

                print(f"min: {mn:.4g}, max: {mx:.4g}")
                print(f"mean: {mean:.4g}, std: {std:.4g}")
                print(f"qtiles: 25%: {p25:.4g} | 50%: {p50:.4g} (med) | 75%: {p75:.4g}")
                
                # Visual Sense of Deviation
                if stats.size > 1:
                    hist = draw_histogram(arr)
                    print(f"dist: |{hist}|")
                    
        except Exception as e:
            # Helpful debugging if stats fail (e.g. on complex numbers)
            print(f"(stats unavailable: {e})")

    def _fmt(x):
        try:
            a = np.asarray(x)
            if args.limit is not None and a.size > args.limit:
                return f"{a.flatten()[: args.limit]} ...({a.size} total)"
            return str(a)
        except Exception:
            return str(x)

    # Print diffs per row if requested
    if args.diff:
        try:
            if np.isscalar(arr) or getattr(arr, "ndim", 0) == 0:
                print("diff: N/A (scalar)")
            elif arr.ndim == 1:
                n = arr.shape[0]
                rows_to_print = n if args.limit is None else min(n, args.limit)
                for i in range(rows_to_print):
                    if i == 0:
                        print(f"idx {i}: value={arr[i]} (diff: N/A)")
                    else:
                        d = arr[i] - arr[i - 1]
                        print(f"idx {i}: value={arr[i]} (diff: {d})")
            else:
                rows = arr.shape[0]
                for i in range(rows):
                    row = arr[i]
                    if i == 0:
                        print(f"row {i}: {_fmt(row)}")
                        print("diff: N/A")
                    else:
                        diff = row - arr[i - 1]
                        print(f"row {i}: {_fmt(row)}")
                        print(f"diff: {_fmt(diff)}")
        except Exception as exc:
            print(f"(could not compute diffs: {exc})")
        print()

    # If limit is set and the array is large, print a flattened preview first
    if args.limit is not None:
        total = getattr(arr, "size", None)
        if total is not None and total > args.limit:
            print(f"(printing first {args.limit} elements of {total})")
            try:
                print(arr.flatten()[: args.limit])
            except Exception:
                print(_fmt(arr))
            print()
            return

    # Default: print the array
    print(arr)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Print contents of .npy files to the console")
    parser.add_argument("files", nargs="+", help="one or more .npy/.npz files to read")
    parser.add_argument("-s", "--summary", action="store_true", help="print a short summary (shape, dtype, stats)")
    parser.add_argument("-d", "--diff", action="store_true", help="print per-row diffs compared to the previous row")
    parser.add_argument("-l", "--limit", type=int, default=None, help="limit number of elements to print")
    args = parser.parse_args()

    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue

        try:
            if path.suffix == ".npz":
                # npz archive can contain multiple arrays
                with np.load(path, allow_pickle=True) as archive:
                    print(f"--- {path} (npz archive) ---")
                    for name in archive.files:
                        arr = archive[name]
                        print(f"== {name} ==")
                        print_array(path.with_name(f"{path.name}:{name}"), arr, args)
                continue

            arr = np.load(path, allow_pickle=True)
            print_array(path, arr, args)

        except Exception as exc:
            print(f"Error loading {path}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())