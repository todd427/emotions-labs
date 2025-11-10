#!/usr/bin/env python3
"""
Enhanced pytest harness generator (v2.3, plot-aware & graphic-safe).

Adds:
- Skips return assertions for plotting/printing functions
- Warns when a graphic may appear
- Forces matplotlib to use a non-interactive backend during tests
"""

import sys
import inspect
import importlib.util
from pathlib import Path
from textwrap import indent


HEADER = """# Auto-generated pytest harness
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import matplotlib
matplotlib.use("Agg")  # Prevent GUI windows during tests
{extra_imports}
import {module_name}

"""

import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

FUNC_TEMPLATE = """def test_{name}({fixture_args}):
    \"\"\"Auto-generated test for {module_name}.{name}\"\"\"
    # TODO: verify outputs and edge cases
{prep_code}    result = {module_name}.{name}({args})
{assertion}
"""

CLASS_TEMPLATE = """class Test{name}:
{methods}
"""

METHOD_TEMPLATE = """    def test_{method}(self{fixture_args}):
        \"\"\"Auto-generated test for {module_name}.{cls}.{method}\"\"\"
{prep_code}        result = {module_name}.{cls}.{method}({args})
{assertion}
"""


def load_module(filepath):
    path = Path(filepath)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def guess_placeholder(param):
    name = param.name.lower()
    if "path" in name or "file" in name:
        return "str(tmp_file)"
    if "text" in name or "string" in name:
        return "'example text'"
    if "df" in name or "dataframe" in name:
        return "pd.DataFrame({'local_date': pd.date_range('2025-01-01', periods=3), 'emotion': ['joy','sadness','joy']})"
    if "emotion" in name:
        return "'joy'"
    if "pred" in name:
        return "[{'label': 'joy', 'score': 0.9}]"
    return "None"


def build_arglist(sig):
    args = []
    use_tmp = False
    use_pd = False
    for p in sig.parameters.values():
        if p.default is not inspect._empty and p.kind == p.POSITIONAL_OR_KEYWORD:
            continue
        val = guess_placeholder(p)
        args.append(f"{p.name}={val}")
        if "tmp_file" in val:
            use_tmp = True
        if "pd." in val:
            use_pd = True
    return ", ".join(args), use_tmp, use_pd


def get_assertion_for(name):
    lowered = name.lower()
    if any(k in lowered for k in ["plot", "show", "display", "print", "savefig"]):
        return (
            "    # Warning: this function may produce a graphic output.\n"
            "    # The matplotlib backend has been set to 'Agg' to prevent windows.\n"
            "    # Ensure the function runs without raising errors.\n"
            "    assert True"
        )
    return "    assert result is not None"


def generate_tests(module_path):
    module = load_module(module_path)
    module_name = Path(module_path).stem
    test_dir = Path("tests")
    test_dir.mkdir(exist_ok=True)
    out_path = test_dir / f"test_{module_name}.py"

    extra_imports = "import pandas as pd"
    code = HEADER.format(module_name=module_name, extra_imports=extra_imports)

    for name, func in inspect.getmembers(module, inspect.isfunction):
        if func.__module__ != module_name:
            continue
        sig = inspect.signature(func)
        args, need_tmp, _ = build_arglist(sig)
        fixture_args = "tmp_path" if need_tmp else ""
        prep_code = ""
        if need_tmp:
            prep_code += "    tmp_file = tmp_path / 'dummy.jsonl'\n    tmp_file.write_text('{}')\n"
        assertion = get_assertion_for(name)
        code += FUNC_TEMPLATE.format(
            name=name,
            module_name=module_name,
            args=args,
            fixture_args=fixture_args,
            prep_code=prep_code,
            assertion=assertion,
        )

    for cls_name, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module_name:
            continue
        methods = ""
        for m_name, method in inspect.getmembers(cls, inspect.isfunction):
            if m_name.startswith("__"):
                continue
            sig = inspect.signature(method)
            args, need_tmp, _ = build_arglist(sig)
            fixture_args = ", tmp_path" if need_tmp else ""
            prep_code = ""
            if need_tmp:
                prep_code += "        tmp_file = tmp_path / 'dummy.jsonl'\n        tmp_file.write_text('{}')\n"
            assertion = get_assertion_for(m_name)
            methods += METHOD_TEMPLATE.format(
                module_name=module_name,
                cls=cls_name,
                method=m_name,
                args=args,
                fixture_args=fixture_args,
                prep_code=prep_code,
                assertion=assertion.replace("\n", "\n        "),
            )
        if methods:
            code += CLASS_TEMPLATE.format(name=cls_name, methods=indent(methods, "    "))

    out_path.write_text(code)
    print(f"✅ Created graphic-safe test harness: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_pytest_harness.py <path/to/module.py>")
        sys.exit(1)
    generate_tests(sys.argv[1])

