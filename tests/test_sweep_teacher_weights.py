# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU-free unit tests for the teacher weight-sweep pure helpers.

Exercises the stdlib-only functions of ``scripts/sweep_teacher_weights.py``
(grid / teacher-pair construction, aggregation, peak detection) without loading
torch / matplotlib / the trainer, so they run on CPU in CI. The module imports
only the standard library and the torch-free ``ensemble_report`` sibling.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "sweep_teacher_weights.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sweep_teacher_weights", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sw = _load_module()


class TestBuildXGrid(unittest.TestCase):
    def test_num_points_inclusive_endpoints(self) -> None:
        grid = sw.build_x_grid(11)
        self.assertEqual(len(grid), 11)
        self.assertEqual(grid[0], 0.0)
        self.assertEqual(grid[-1], 1.0)
        self.assertEqual(grid, sorted(grid))
        self.assertEqual(sw.build_x_grid(5), [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_x_values_override_dedup_sorted(self) -> None:
        self.assertEqual(sw.build_x_grid(11, [0.7, 0.3, 0.3]), [0.3, 0.7])

    def test_rounding_to_three_decimals(self) -> None:
        # 1/3 and 2/3 round to 3 decimals (matches x<X.XXX> dir naming).
        grid = sw.build_x_grid(4)
        self.assertEqual(grid, [0.0, 0.333, 0.667, 1.0])

    def test_rejects_too_few_points(self) -> None:
        with self.assertRaises(ValueError):
            sw.build_x_grid(1)

    def test_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            sw.build_x_grid(11, [0.5, 1.5])


class TestResolveTeachers(unittest.TestCase):
    LABELS = ["OCR-Teacher", "Pickscore-Teacher", "GenEval-Teacher"]

    def test_none_selects_all(self) -> None:
        self.assertEqual(sw.resolve_teachers(None, self.LABELS), self.LABELS)

    def test_by_label(self) -> None:
        self.assertEqual(
            sw.resolve_teachers(["OCR-Teacher", "GenEval-Teacher"], self.LABELS),
            ["OCR-Teacher", "GenEval-Teacher"],
        )

    def test_by_index(self) -> None:
        self.assertEqual(
            sw.resolve_teachers(["0", "2"], self.LABELS), ["OCR-Teacher", "GenEval-Teacher"]
        )

    def test_dedup(self) -> None:
        self.assertEqual(
            sw.resolve_teachers(["OCR-Teacher", "0", "GenEval-Teacher"], self.LABELS),
            ["OCR-Teacher", "GenEval-Teacher"],
        )

    def test_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            sw.resolve_teachers(["nope"], self.LABELS)

    def test_rejects_fewer_than_two(self) -> None:
        with self.assertRaises(ValueError):
            sw.resolve_teachers(["OCR-Teacher", "OCR-Teacher"], self.LABELS)


class TestPairs(unittest.TestCase):
    def test_all_two_combinations(self) -> None:
        self.assertEqual(
            sw.teacher_pairs(["A", "B", "C"]), [("A", "B"), ("A", "C"), ("B", "C")]
        )

    def test_pair_label_roundtrip(self) -> None:
        label = sw.pair_label("OCR-Teacher", "GenEval-Teacher")
        self.assertEqual(label, "OCR-Teacher__GenEval-Teacher")
        self.assertEqual(sw._split_pair(label), ("OCR-Teacher", "GenEval-Teacher"))

    def test_x_dirname(self) -> None:
        self.assertEqual(sw.x_dirname(0.3), "x0.300")
        self.assertEqual(sw.x_dirname(1.0), "x1.000")


class TestAggregateSweep(unittest.TestCase):
    def _records(self):
        return [
            {"pair": "A__B", "test_set": "ocr", "x": 0.0, "gidx": 0, "scores": {"ocr": 0.2}},
            {"pair": "A__B", "test_set": "ocr", "x": 0.0, "gidx": 1, "scores": {"ocr": 0.4}},
            {"pair": "A__B", "test_set": "ocr", "x": 1.0, "gidx": 0, "scores": {"ocr": 0.9}},
            {"pair": "A__B", "test_set": "ocr", "x": 1.0, "gidx": 1, "scores": {"ocr": 0.7}},
        ]

    def test_mean_per_x(self) -> None:
        agg = sw.aggregate_sweep(self._records())
        curve = agg["A__B"]["ocr"]["ocr"]
        self.assertAlmostEqual(curve[0.0]["mean"], 0.3)
        self.assertAlmostEqual(curve[1.0]["mean"], 0.8)
        self.assertEqual(curve[0.0]["n"], 2)


class TestFindPeaks(unittest.TestCase):
    def _agg(self, means_by_x):
        return {
            "A__B": {
                "ocr": {"ocr": {x: {"mean": m, "std": 0.0, "n": 1} for x, m in means_by_x.items()}}
            }
        }

    def test_argmax(self) -> None:
        peaks = sw.find_peaks(self._agg({0.0: 0.2, 0.5: 0.9, 1.0: 0.4}))
        info = peaks["A__B"]["ocr"]["ocr"]
        self.assertEqual(info["peak_x"], 0.5)
        self.assertAlmostEqual(info["peak_value"], 0.9)
        self.assertAlmostEqual(info["value_at_min"], 0.2)
        self.assertAlmostEqual(info["value_at_max"], 0.4)

    def test_tie_breaks_to_smaller_x(self) -> None:
        peaks = sw.find_peaks(self._agg({0.2: 0.5, 0.5: 0.3, 0.8: 0.5}))
        self.assertEqual(peaks["A__B"]["ocr"]["ocr"]["peak_x"], 0.2)

    def test_gallery_x_values_dedup(self) -> None:
        peaks = sw.find_peaks(self._agg({0.0: 0.2, 0.5: 0.9, 1.0: 0.4}))
        xs = sw._gallery_x_values(peaks["A__B"]["ocr"], 0.0, 1.0)
        self.assertEqual(xs, [0.0, 0.5, 1.0])


class TestLoadCachedRecordsSweep(unittest.TestCase):
    """The shared loader must accept the sweep record schema via a custom identity."""

    _SWEEP_KEY = staticmethod(lambda r: (r["pair"], r["test_set"], r["x"], int(r["gidx"])))

    def test_loads_sweep_records_without_method_field(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            recs = [
                {"pair": "A__B", "test_set": "ocr", "x": 0.0, "gidx": 0, "scores": {"ocr": 0.2}},
                {"pair": "A__B", "test_set": "ocr", "x": 0.5, "gidx": 0, "scores": {"ocr": 0.9}},
            ]
            sw._write_record_shard(out / "records", "ocr", "A__B_x0.000", 0, recs)
            loaded = sw._load_cached_records(out, identity=self._SWEEP_KEY)
            # no KeyError on the missing "method" field; both records returned
            self.assertEqual(len(loaded), 2)
            self.assertEqual({r["x"] for r in loaded}, {0.0, 0.5})

    def test_newer_shard_wins_per_identity(self) -> None:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rdir = out / "records"
            sw._write_record_shard(
                rdir, "ocr", "A__B_x0.000__r0", 0,
                [{"pair": "A__B", "test_set": "ocr", "x": 0.0, "gidx": 0, "scores": {"ocr": 0.1}}],
            )
            # a later (newer mtime) shard re-scores the same (pair, test_set, x, gidx)
            new_shard = rdir / "ocr__A__B_x0.000__r1.jsonl"
            new_shard.write_text(
                '{"pair": "A__B", "test_set": "ocr", "x": 0.0, "gidx": 0, "scores": {"ocr": 0.9}}\n',
                encoding="utf-8",
            )
            os.utime(new_shard, (10**10, 10**10))  # force newest mtime
            loaded = sw._load_cached_records(out, identity=self._SWEEP_KEY)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["scores"]["ocr"], 0.9)


if __name__ == "__main__":
    unittest.main()
