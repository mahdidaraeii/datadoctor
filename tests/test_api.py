import dataclasses
import importlib
import inspect

import pytest

import datadoctor

PUBLIC_API = {
    "AnalysisConfig": "datadoctor.core.config",
    "AnalysisResult": "datadoctor.core.result",
    "Dataset": "datadoctor.core.dataset",
    "Finding": "datadoctor.core.result",
    "Severity": "datadoctor.core.result",
}


@pytest.mark.parametrize("name", sorted(PUBLIC_API))
def test_name_is_importable_from_the_package_root(name):
    namespace: dict = {}
    exec(f"from datadoctor import {name}", namespace)

    original = getattr(importlib.import_module(PUBLIC_API[name]), name)
    assert namespace[name] is original


def test_all_lists_exactly_the_public_api():
    assert sorted(datadoctor.__all__) == sorted(PUBLIC_API)


def test_star_import_yields_exactly_the_public_api():
    namespace: dict = {}
    exec("from datadoctor import *", namespace)

    exported = {name for name in namespace if name != "__builtins__"}
    assert exported == set(PUBLIC_API)


def test_version_is_still_available_as_an_attribute():
    assert isinstance(datadoctor.__version__, str) and datadoctor.__version__


def _public_members(cls):
    for name, member in inspect.getmembers(cls):
        if not name.startswith("_") and (inspect.isroutine(member) or isinstance(member, property)):
            yield name, member


@pytest.mark.parametrize("name", sorted(PUBLIC_API))
class TestDocumentation:
    def test_class_has_a_docstring(self, name):
        assert inspect.getdoc(getattr(datadoctor, name))

    def test_public_methods_and_properties_have_docstrings(self, name):
        cls = getattr(datadoctor, name)

        undocumented = [
            member_name
            for member_name, member in _public_members(cls)
            if not inspect.getdoc(getattr(cls, member_name, member))
        ]

        assert undocumented == []


DATACLASS_NAMES = [
    n for n in sorted(PUBLIC_API) if dataclasses.is_dataclass(getattr(datadoctor, n))
]


@pytest.mark.parametrize("name", DATACLASS_NAMES)
def test_docstring_mentions_every_dataclass_field(name):
    cls = getattr(datadoctor, name)

    doc = inspect.getdoc(cls)
    undocumented = [f.name for f in dataclasses.fields(cls) if f.name not in doc]

    assert undocumented == []
