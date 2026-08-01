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

"""Contract and CPU execution tests for the Mixture-of-Density notebook."""

import ast
import json
import math
import os
import re
import time
from importlib.util import find_spec
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "docs" / "xopd" / "demos" / "mixture_of_density_demo.ipynb"
TARGET_NOTEBOOK_PATTERN = "!docs/xopd/demos/mixture_of_density_demo.ipynb"
FULL_TEST_COMMAND = (
    "uv run --isolated --frozen --with pytest --with matplotlib "
    "pytest tests/test_mixture_of_density_demo.py"
)
OPTIONAL_VIZ_MODULES = ("matplotlib", "PIL", "imageio", "imageio_ffmpeg")


def _load_notebook():
    with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _cell_source(cell):
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def _code_sources(notebook):
    return [_cell_source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"]


def _assert_finite_tree(value, path="root"):
    if value is None or isinstance(value, (str, bool, Path)):
        return
    if isinstance(value, (int, float)):
        assert math.isfinite(float(value)), f"{path} is non-finite: {value!r}"
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_tree(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{path}[{index}]")
        return
    raise AssertionError(f"{path} has unsupported summary type {type(value).__name__}")


def _attribute_path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _static_string(node, assignments):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        assigned = assignments.get(node.id)
        return _static_string(assigned, assignments) if assigned is not None else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left, assignments)
        right = _static_string(node.right, assignments)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Constant):
                parts.append(str(value.value.value))
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.Call) and _attribute_path(node.func) == "torch.device" and node.args:
        return _static_string(node.args[0], assignments)
    return None


def _assert_cpu_only_source(source, filename):
    tree = ast.parse(source, filename=filename)
    assignments = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_path = _attribute_path(node.func)
        assert (
            function_path != "torch.set_default_device"
        ), f"torch.set_default_device is forbidden in {filename}"
        assert not function_path.endswith(".cuda"), f"CUDA call found in {filename}"
        assert ".cuda." not in f".{function_path}.", f"CUDA API found in {filename}"
        assert ".mps." not in f".{function_path}.", f"MPS API found in {filename}"

        device_nodes = []
        if function_path == "torch.device" and node.args:
            device_nodes.append(node.args[0])
        if function_path.endswith(".to") and node.args:
            device_nodes.append(node.args[0])
        device_nodes.extend(keyword.value for keyword in node.keywords if keyword.arg == "device")
        for device_node in device_nodes:
            static_device = _static_string(device_node, assignments)
            if static_device is None:
                continue
            normalized_device = static_device.lower()
            assert (
                "cuda" not in normalized_device
            ), f"CUDA device {static_device!r} found in {filename}"
            assert (
                "mps" not in normalized_device
            ), f"MPS device {static_device!r} found in {filename}"


def _require_optional_viz_dependencies():
    missing = [module for module in OPTIONAL_VIZ_MODULES if find_spec(module) is None]
    if missing:
        pytest.skip(
            "Mixture-of-Density execution requires optional visualization dependencies "
            f"{missing!r}. Run `{FULL_TEST_COMMAND}`."
        )


def _execute_notebook(work_directory, test_mode, monkeypatch):
    _require_optional_viz_dependencies()
    if test_mode:
        monkeypatch.setenv("MIXTURE_DENSITY_TEST_MODE", "1")
    else:
        monkeypatch.delenv("MIXTURE_DENSITY_TEST_MODE", raising=False)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.chdir(work_directory)

    namespace = {"__name__": "__mixture_density_notebook_test__"}
    started_at = time.perf_counter()
    notebook = _load_notebook()
    for cell_index, source in enumerate(_code_sources(notebook)):
        compiled = compile(
            source,
            filename=f"{NOTEBOOK_PATH.name}:cell-{cell_index}",
            mode="exec",
        )
        exec(compiled, namespace)
    namespace["_MEASURED_TOTAL_RUNTIME_SECONDS"] = time.perf_counter() - started_at
    return namespace


def test_notebook_structure_markers_and_clean_state():
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    assert isinstance(notebook.get("nbformat_minor"), int)
    assert isinstance(notebook.get("cells"), list) and notebook["cells"]

    markdown = "\n".join(
        _cell_source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "markdown"
    )
    code = "\n".join(_code_sources(notebook))
    required_headings = (
        "# Mixture-of-Density: a CPU-only visual tutorial",
        "## Density diagnostics, metrics, and animations",
        "## Conditional 8×8 image analogy",
        "## Flow-Factory mapping and practical run guide",
    )
    for heading in required_headings:
        assert heading in markdown
    assert "Todo 3" not in markdown
    assert "Todo 4" not in markdown

    required_code_markers = (
        "MIXTURE_DENSITY_TEST_MODE",
        "RUN_2D_TRAINING = not MIXTURE_DENSITY_TEST_MODE",
        "select_final_best_evaluation",
        "FINAL_BEST_2D_EVALUATION",
        "ANIMATION_2D_EVALUATIONS",
        "SMOKE_2D_SUMMARY",
        "VISUAL_SMOKE_SUMMARY",
        "TINY_IMAGE_SMOKE_SUMMARY",
        "TINY_IMAGE_METRIC_RESOLUTION_SUMMARY",
        "NOTEBOOK_RUNTIME_SUMMARY",
        "wrong constant arithmetic velocity average",
        "wrong per-step branch redraw/switching",
    )
    for marker in required_code_markers:
        assert marker in code

    for cell_index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None, f"cell {cell_index} has execution_count"
        assert cell.get("outputs") == [], f"cell {cell_index} has saved outputs"
        ast.parse(_cell_source(cell), filename=f"{NOTEBOOK_PATH.name}:cell-{cell_index}")


def test_core_contract_remains_active_without_optional_viz():
    notebook = _load_notebook()
    code = "\n".join(_code_sources(notebook))
    markdown = "\n".join(
        _cell_source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "markdown"
    )
    assert FULL_TEST_COMMAND in markdown
    assert "--extra viz" not in markdown
    assert "--extra viz" not in code
    assert "CHECKPOINT_2D_EVALUATIONS[-1]" not in code
    assert "smoke_evaluations[-1]" not in code
    assert 'DEVICE = torch.device("cpu")' in code


def test_gitignore_tracks_only_the_demo_notebook():
    gitignore_lines = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    notebook_rules = [
        line.strip()
        for line in gitignore_lines
        if line.strip() and not line.lstrip().startswith("#") and "ipynb" in line
    ]
    assert "*.ipynb" in notebook_rules
    assert ".ipynb_checkpoints/" in notebook_rules
    exceptions = [rule for rule in notebook_rules if rule.startswith("!")]
    assert exceptions == [TARGET_NOTEBOOK_PATTERN]


@pytest.mark.parametrize(
    "source",
    (
        "tensor.cuda()",
        'tensor.to("cuda:0")',
        'torch.zeros(2, device="cuda")',
        'tensor.to(device=torch.device("mps"))',
        'torch.set_default_device("cpu")',
        'accelerator = "cu" + "da"; tensor.to(accelerator)',
        'torch.device(f"cuda:{0}")',
    ),
)
def test_cpu_guard_rejects_accelerator_construction(source):
    with pytest.raises(AssertionError):
        _assert_cpu_only_source(source, "<accelerator-probe>")


def test_notebook_has_no_accelerator_device_or_call():
    notebook = _load_notebook()
    for cell_index, source in enumerate(_code_sources(notebook)):
        _assert_cpu_only_source(source, f"{NOTEBOOK_PATH.name}:code-{cell_index}")
        assert re.search(r"\.cuda\s*\(", source) is None


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)
def test_cpu_test_mode_executes_all_smoke_paths(monkeypatch, tmp_path):
    namespace = _execute_notebook(tmp_path, test_mode=True, monkeypatch=monkeypatch)
    torch = namespace["torch"]
    np = namespace["np"]

    assert namespace["MIXTURE_DENSITY_TEST_MODE"] is True
    assert namespace["DEVICE"].type == "cpu"
    assert namespace["RUN_2D_TRAINING"] is False
    assert namespace["TRAINING_2D_ARTIFACTS"] is None

    core = namespace["SMOKE_2D_SUMMARY"]
    visual = namespace["VISUAL_SMOKE_SUMMARY"]
    tiny = namespace["TINY_IMAGE_SMOKE_SUMMARY"]
    resolution = namespace["TINY_IMAGE_METRIC_RESOLUTION_SUMMARY"]
    for name, summary in (
        ("core", core),
        ("visual", visual),
        ("tiny", tiny),
        ("resolution", resolution),
    ):
        assert summary is not None, f"{name} summary was not produced"
        _assert_finite_tree(summary, name)

    assert core["finite"] and core["same_seed_reproduced"]
    assert core["teacher_branches"] > 0 and core["old_student_branches"] > 0
    assert visual["frame_count"] == 1
    assert visual["frame_dtype"] == "uint8"
    assert len(visual["frame_shape"]) == 3 and visual["frame_shape"][-1] == 3

    visual_gif = Path(visual["paths"]["gif"])
    assert visual_gif.is_file() and visual_gif.stat().st_size > 0
    visual_timing = visual["gif_timing"]
    assert visual_timing["frame_count"] == 1

    assert tiny["finite"] and tiny["finite_gradients"]
    assert tiny["condition_counts"] == [32, 32, 32]
    assert tiny["teacher_branches"] > 0 and tiny["old_student_branches"] > 0
    assert all(count > 0 for count in tiny["source_branch_condition_counts"].values())
    assert all(count > 0 for count in tiny["target_branch_condition_counts"].values())
    assert tiny["frame_shape"] == (480, 640, 3)
    assert resolution["rare_mode_resolvable"] == [True, True, True]

    repeat_kwargs = {
        "teacher": namespace["TINY_IMAGE_TEACHER"],
        "alpha": namespace["CONFIG"].alpha,
        "outer_iterations": 1,
        "cache_size": 48,
        "inner_updates": 1,
        "batch_size": 12,
        "ode_steps": 2,
        "student_width": 16,
        "learning_rate": namespace["CONFIG"].learning_rate,
        "seed": namespace["CONFIG"].seed + 90_000,
        "metric_samples_per_condition": 16,
        "display_samples_per_condition": 16,
        "export_gif": False,
    }
    first = namespace["run_tiny_image_outer_loop"](**repeat_kwargs)
    second = namespace["run_tiny_image_outer_loop"](**repeat_kwargs)
    assert first.records[0].losses == second.records[0].losses
    assert torch.equal(first.records[0].actual_endpoints, second.records[0].actual_endpoints)
    assert torch.equal(first.records[0].target_endpoints, second.records[0].target_endpoints)
    assert first.records[0].metrics == second.records[0].metrics
    assert first.records[0].frame.dtype == np.uint8


@pytest.fixture(scope="session")
def normal_fast_mode_execution():
    work_directory = REPOSITORY_ROOT
    monkeypatch = pytest.MonkeyPatch()
    try:
        namespace = _execute_notebook(
            work_directory,
            test_mode=False,
            monkeypatch=monkeypatch,
        )
    finally:
        monkeypatch.undo()
    return namespace, work_directory


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)
def test_normal_fast_mode_executes_once_with_final_best_artifacts(
    normal_fast_mode_execution,
):
    namespace, work_directory = normal_fast_mode_execution
    torch = namespace["torch"]
    config = namespace["CONFIG"]
    artifacts = namespace["TRAINING_2D_ARTIFACTS"]
    final_record = namespace["OUTER_2D_RESULT"].iterations[-1]
    final_best = namespace["FINAL_BEST_2D_EVALUATION"]

    assert namespace["MIXTURE_DENSITY_TEST_MODE"] is False
    assert namespace["RUN_2D_TRAINING"] is True
    assert namespace["DEVICE"].type == "cpu"
    assert artifacts is not None
    assert int(final_best["outer_iteration"]) == final_record.outer_iteration
    assert int(final_best["inner_update"]) == final_record.best_update
    assert final_best["is_selected_best"] is True
    assert namespace["ANIMATION_2D_EVALUATIONS"][-1] is final_best
    assert len(namespace["ANIMATION_2D_EVALUATIONS"]) == len(namespace["CHECKPOINT_2D_EVALUATIONS"])
    _assert_finite_tree(final_best["metrics"], "final_best.metrics")
    assert all(
        parameter.device.type == "cpu"
        for parameter in namespace["OUTER_2D_RESULT"].final_student.parameters()
    )
    assert final_best["endpoint_samples"].device.type == "cpu"

    expected_frame_duration = 1.0 / config.export_fps
    assert artifacts["outer_frame_count"] == artifacts["checkpoint_count"]
    assert artifacts["generation_frame_count"] == config.ode_steps + 1
    for timing_key, frame_count_key in (
        ("outer_gif_timing", "outer_frame_count"),
        ("generation_gif_timing", "generation_frame_count"),
    ):
        timing = artifacts[timing_key]
        expected_frame_count = artifacts[frame_count_key]
        assert timing["frame_count"] == expected_frame_count
        assert timing["timing_available"] is True
        assert timing["mean_frame_duration_seconds"] == pytest.approx(
            expected_frame_duration,
            abs=0.02,
        )

    artifact_paths = [
        *artifacts["outer_training_paths"].values(),
        *artifacts["generation_trajectory_paths"].values(),
        artifacts["wrong_control_path"],
        artifacts["matched_kde_responsibility_path"],
        artifacts["metrics_path"],
        artifacts["runtime_summary_path"],
    ]
    assert set(artifacts["outer_training_paths"]) == {"gif", "mp4"}
    assert set(artifacts["generation_trajectory_paths"]) == {"gif", "mp4"}
    for raw_path in artifact_paths:
        path = work_directory / raw_path
        assert path.is_file(), f"missing normal-mode artifact: {path}"
        assert path.stat().st_size > 0, f"empty normal-mode artifact: {path}"

    metrics_payload = json.loads(
        (work_directory / artifacts["metrics_path"]).read_text(encoding="utf-8")
    )
    persisted_final_best = metrics_payload["final_best_evaluation"]
    assert persisted_final_best["outer_iteration"] == final_record.outer_iteration
    assert persisted_final_best["inner_update"] == final_record.best_update
    _assert_finite_tree(persisted_final_best["metrics"], "persisted_final_best.metrics")

    runtime_summary = namespace["NOTEBOOK_RUNTIME_SUMMARY"]
    assert runtime_summary["main_2d_runtime_seconds"] == pytest.approx(artifacts["runtime_seconds"])
    assert runtime_summary["total_notebook_runtime_seconds"] >= artifacts["runtime_seconds"]
    assert (
        namespace["_MEASURED_TOTAL_RUNTIME_SECONDS"]
        >= runtime_summary["total_notebook_runtime_seconds"]
    )
    assert torch.equal(
        final_best["endpoint_samples"],
        final_best["trajectory"][1][:, -1],
    )


def test_normal_mode_defaults_to_full_2d_training(monkeypatch):
    monkeypatch.delenv("MIXTURE_DENSITY_TEST_MODE", raising=False)
    notebook = _load_notebook()
    config_source = next(
        source for source in _code_sources(notebook) if "class DemoConfig" in source
    )
    config_tree = ast.parse(config_source)
    assignment = next(
        node
        for node in ast.walk(config_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MIXTURE_DENSITY_TEST_MODE"
            for target in node.targets
        )
    )
    namespace = {"os": os}
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), "<test-mode-assignment>", "exec"),
        namespace,
    )
    assert namespace["MIXTURE_DENSITY_TEST_MODE"] is False
