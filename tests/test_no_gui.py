import ast
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "src" / "datadoctor"


def test_no_module_imports_pyplot_or_shows_a_figure():
    """Plots are files. Nothing may open a window, so no code reaches for pyplot or show()."""
    offenders = []
    for path in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import) and any("pyplot" in a.name for a in node.names):
                offenders.append(f"{path.name}:{node.lineno} imports pyplot")
            if isinstance(node, ast.ImportFrom) and (
                "pyplot" in (node.module or "") or any(a.name == "pyplot" for a in node.names)
            ):
                offenders.append(f"{path.name}:{node.lineno} imports pyplot")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "show"
            ):
                offenders.append(f"{path.name}:{node.lineno} calls show()")

    assert offenders == []
