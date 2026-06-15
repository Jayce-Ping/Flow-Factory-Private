# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""GPU-free unit tests for the ensemble-methods comparison report helpers.

Exercises the pure functions of ``scripts/compare_ensemble_methods.py``
(``aggregate_metrics``, ``render_html``, ``load_method_specs``) without loading
the model/trainer, so they run on CPU in CI.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "compare_ensemble_methods.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_ensemble_methods", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection (cls.__module__) resolves.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cmp = _load_module()


class TestAggregateMetrics(unittest.TestCase):
    def _records(self):
        return [
            {
                "test_set": "ocr",
                "method": "A",
                "gidx": 0,
                "scores": {"ocr": 0.8, "pick_score": 0.3},
                "tag": None,
            },
            {
                "test_set": "ocr",
                "method": "A",
                "gidx": 1,
                "scores": {"ocr": 0.6, "pick_score": 0.5},
                "tag": None,
            },
            {
                "test_set": "ocr",
                "method": "B",
                "gidx": 0,
                "scores": {"ocr": 0.2, "pick_score": 0.9},
                "tag": None,
            },
            {
                "test_set": "geneval",
                "method": "A",
                "gidx": 0,
                "scores": {"geneval": 1.0},
                "tag": "color",
            },
            {
                "test_set": "geneval",
                "method": "A",
                "gidx": 1,
                "scores": {"geneval": 0.0},
                "tag": "count",
            },
        ]

    def test_mean_std_n(self) -> None:
        summary = cmp.aggregate_metrics(self._records())
        a = summary["aggregate"]["ocr"]["ocr"]["A"]
        self.assertAlmostEqual(a["mean"], 0.7)
        self.assertAlmostEqual(a["std"], 0.1)  # population std of [0.8, 0.6]
        self.assertEqual(a["n"], 2)
        b = summary["aggregate"]["ocr"]["ocr"]["B"]
        self.assertAlmostEqual(b["mean"], 0.2)
        self.assertEqual(b["n"], 1)

    def test_per_tag(self) -> None:
        summary = cmp.aggregate_metrics(self._records())
        per_tag = summary["per_tag"]["geneval"]["geneval"]
        self.assertAlmostEqual(per_tag["color"]["A"]["mean"], 1.0)
        self.assertAlmostEqual(per_tag["count"]["A"]["mean"], 0.0)

    def test_empty_records(self) -> None:
        summary = cmp.aggregate_metrics([])
        self.assertEqual(summary["aggregate"], {})
        self.assertEqual(summary["per_tag"], {})

    def test_mean_std_handles_nan_and_empty(self) -> None:
        self.assertEqual(cmp._mean_std([])["n"], 0)
        self.assertTrue(math.isnan(cmp._mean_std([])["mean"]))
        out = cmp._mean_std([float("nan"), 1.0, 3.0])
        self.assertAlmostEqual(out["mean"], 2.0)
        self.assertEqual(out["n"], 2)


class TestRenderHtml(unittest.TestCase):
    def _summary(self):
        return cmp.aggregate_metrics(
            [
                {"test_set": "ocr", "method": "A", "gidx": 0, "scores": {"ocr": 0.8}, "tag": None},
                {"test_set": "ocr", "method": "B", "gidx": 0, "scores": {"ocr": 0.2}, "tag": None},
            ]
        )

    def _meta(self):
        return {
            "methods": ["A", "B"],
            "test_sets": ["ocr"],
            "seed": 42,
            "num_inference_steps": 40,
            "guidance_scale": 4.5,
            "resolution": 512,
            "num_prompts_per_set": {"ocr": 1},
        }

    def _gallery(self):
        return {
            "ocr": [
                {
                    "gidx": 0,
                    "prompt": 'a sign saying "Hi" <b>',
                    "tag": None,
                    "include": None,
                    "methods": {
                        "A": {"img": "images/ocr/A/00000.png", "scores": {"ocr": 0.8}},
                        "B": {"img": None, "scores": {"ocr": 0.2}},
                    },
                }
            ]
        }

    def test_html_structure(self) -> None:
        out = cmp.render_html(self._summary(), self._gallery(), self._meta())
        self.assertIn("<!doctype html>", out)
        self.assertIn("</html>", out)
        # both methods appear
        self.assertIn(">A<", out)
        self.assertIn(">B<", out)
        # image src + missing placeholder
        self.assertIn("images/ocr/A/00000.png", out)
        self.assertIn("no image", out)
        # best-mean highlight present (A=0.8 beats B=0.2)
        self.assertIn('class="best"', out)
        # lazy loading
        self.assertIn("loading='lazy'", out)

    def test_html_escapes_prompt(self) -> None:
        out = cmp.render_html(self._summary(), self._gallery(), self._meta())
        # The prompt's raw "<b>" must be HTML-escaped (the template itself uses
        # <b> for the index, so we check the escaped form + absence of the raw run).
        self.assertIn("&lt;b&gt;", out)
        self.assertNotIn('saying "Hi" <b>', out)

    def _summary_with_baseline(self):
        return cmp.aggregate_metrics(
            [
                # base baseline (weak), blend A (best fusion), blend B
                {
                    "test_set": "ocr",
                    "method": "base",
                    "gidx": 0,
                    "scores": {"ocr": 0.1},
                    "tag": None,
                },
                {"test_set": "ocr", "method": "A", "gidx": 0, "scores": {"ocr": 0.8}, "tag": None},
                {"test_set": "ocr", "method": "B", "gidx": 0, "scores": {"ocr": 0.2}, "tag": None},
            ]
        )

    def _meta_with_baseline(self):
        meta = self._meta()
        meta["methods"] = ["base", "A", "B"]
        meta["baseline_methods"] = ["base"]
        return meta

    def _gallery_with_baseline(self):
        return {
            "ocr": [
                {
                    "gidx": 0,
                    "prompt": "p",
                    "tag": None,
                    "include": None,
                    "methods": {
                        "base": {"img": "images/ocr/base/00000.png", "scores": {"ocr": 0.1}},
                        "A": {"img": "images/ocr/A/00000.png", "scores": {"ocr": 0.8}},
                        "B": {"img": None, "scores": {"ocr": 0.2}},
                    },
                }
            ]
        }

    def test_html_baseline_styling(self) -> None:
        out = cmp.render_html(
            self._summary_with_baseline(),
            self._gallery_with_baseline(),
            self._meta_with_baseline(),
        )
        # baseline row + ref badge + legend present
        self.assertIn("baseline", out)
        self.assertIn(">ref<", out)
        self.assertIn("class='legend'", out)
        # best fusion method (A) highlighted green; best baseline (base) brown
        self.assertIn('class="best"', out)
        self.assertIn('class="basebest"', out)
        # gallery separator between reference and fusion columns
        self.assertIn("class='sep'", out)


class TestLoadMethodSpecs(unittest.TestCase):
    def test_loads_ablation_configs(self) -> None:
        glob_pat = str(_REPO_ROOT / "ensemble-eval/lora/sd3_5/ablations" / "*.yaml")
        specs = cmp.load_method_specs(glob_pat)
        labels = {s.label for s in specs}
        # the kl/kl_inv variants and the uniform baselines are present
        self.assertIn("3_geneval-ocr-pickscore_pcgrad_residual_kl", labels)
        self.assertIn("3_geneval-ocr-pickscore_pcgrad_residual_kl_inv", labels)
        self.assertIn("3_geneval-ocr-pickscore_weighted", labels)
        by_label = {s.label: s for s in specs}
        kl = by_label["3_geneval-ocr-pickscore_pcgrad_residual_kl"]
        self.assertEqual(kl.blend_mode, "pcgrad_residual")
        self.assertEqual(kl.weighting, "kl")
        kl_inv = by_label["3_geneval-ocr-pickscore_pcgrad_residual_kl_inv"]
        self.assertEqual(kl_inv.weighting, "kl_inv")
        # labels are unique
        self.assertEqual(len(labels), len(specs))

    def test_missing_glob_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            cmp.load_method_specs(str(_REPO_ROOT / "no_such_dir" / "*.yaml"))

    def test_build_baseline_specs(self) -> None:
        names = ["eval_ckpt_0", "eval_ckpt_1"]
        paths = ["Org/OCR-Teacher", "Org/GenEval-Teacher"]
        specs = cmp.build_baseline_specs(names, paths)
        self.assertEqual(
            [s.label for s in specs],
            ["baseline_base", "baseline_OCR-Teacher", "baseline_GenEval-Teacher"],
        )
        self.assertEqual(specs[0].kind, "base")
        self.assertIsNone(specs[0].checkpoint_name)
        self.assertEqual(specs[1].kind, "single")
        self.assertEqual(specs[1].checkpoint_name, "eval_ckpt_0")
        self.assertEqual(specs[2].checkpoint_name, "eval_ckpt_1")
        # Every baseline runs as a plain weighted forward (no PCGrad/TIES).
        for spec in specs:
            self.assertEqual(spec.blend_mode, "weighted")

    def test_build_baseline_specs_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            cmp.build_baseline_specs(["eval_ckpt_0"], ["a", "b"])

    def test_default_methods_resolve(self) -> None:
        glob_pat = str(_REPO_ROOT / "ensemble-eval/lora/sd3_5/ablations" / "*.yaml")
        specs = cmp.load_method_specs(glob_pat)
        by_label = {s.label: s for s in specs}
        self.assertEqual(
            cmp.DEFAULT_METHODS,
            (
                "3_geneval-ocr-pickscore_weighted",
                "3_geneval-ocr-pickscore_pcgrad_residual",
                "3_geneval-ocr-pickscore_pcgrad_residual_channelwise",
                "3_geneval-ocr-pickscore_pcgrad_residual_normalized",
                "3_geneval-ocr-pickscore_pcgrad_residual_kl",
                "3_geneval-ocr-pickscore_pcgrad_residual_kl_inv",
                "3_geneval-ocr-pickscore_ties",
            ),
        )
        for label in cmp.DEFAULT_METHODS:
            self.assertIn(label, by_label, f"default method {label!r} missing from ablations")
        # 'weighted' is the linear-blend baseline; the rest are base-anchored.
        self.assertEqual(by_label["3_geneval-ocr-pickscore_weighted"].blend_mode, "weighted")
        for label in cmp.DEFAULT_METHODS[1:]:
            mode = by_label[label].blend_mode
            self.assertTrue(
                mode.startswith("pcgrad_residual") or mode == "ties",
                f"default method {label!r} has non-base-anchored mode {mode!r}",
            )
        # The kl / kl_inv variants carry the dynamic weighting flags.
        self.assertEqual(by_label["3_geneval-ocr-pickscore_pcgrad_residual_kl"].weighting, "kl")
        self.assertEqual(
            by_label["3_geneval-ocr-pickscore_pcgrad_residual_kl_inv"].weighting, "kl_inv"
        )


if __name__ == "__main__":
    unittest.main()
