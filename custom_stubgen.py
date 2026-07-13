#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Dict, Optional, Iterable
from collections import defaultdict

from pybind11_stubgen import (
    Writer,
    QualifiedName,
    Printer,
    arg_parser,
    stub_parser_from_args,
    to_output_and_subdir,
    run,
)
from pybind11_stubgen.parser.mixins.fix import FixCurrentModulePrefixInTypeNames
from pybind11_stubgen.structs import Function, Import, ResolvedType, Module

# pybind11-stubgen only strips the current module prefix from the top-level type name,
# missing union members (e.g. "_franky.Affine | None"). Patch it to strip recursively.
_orig_parse_annotation_str = FixCurrentModulePrefixInTypeNames.parse_annotation_str


def _strip_current_module_recursive(parser, annotation):
    if isinstance(annotation, ResolvedType):
        annotation.name = parser._strip_current_module(annotation.name)
        for param in annotation.parameters or []:
            _strip_current_module_recursive(parser, param)


def _parse_annotation_str(self, annotation_str):
    result = _orig_parse_annotation_str(self, annotation_str)
    _strip_current_module_recursive(self, result)
    return result


FixCurrentModulePrefixInTypeNames.parse_annotation_str = _parse_annotation_str


class CustomWriter(Writer):
    def __init__(
        self, alternative_types: Dict[str, Iterable[str]], stub_ext: str = "pyi"
    ):
        super().__init__(stub_ext=stub_ext)
        self.alternative_types = {
            QualifiedName.from_str(k): tuple(QualifiedName.from_str(e) for e in v)
            for k, v in alternative_types.items()
        }

    def _patch_function(self, function: Function):
        for argument in function.args:
            if (
                argument.annotation is not None
                and argument.annotation.name in self.alternative_types
            ):
                converted_types = [
                    ResolvedType(e)
                    for e in self.alternative_types[argument.annotation.name]
                ]
                argument.annotation = ResolvedType(
                    QualifiedName.from_str("typing.Union"),
                    [argument.annotation] + converted_types,
                )
        self._fix_implicit_optional(function)

    @staticmethod
    def _fix_implicit_optional(function: Function):
        # pybind11 renders nullable pointer args as "arg: T = None"; make them Optional
        for argument in function.args:
            if (
                argument.default is not None
                and str(argument.default) == "None"
                and isinstance(argument.annotation, ResolvedType)
                and argument.annotation.name
                not in (
                    QualifiedName.from_str("typing.Union"),
                    QualifiedName.from_str("typing.Optional"),
                )
            ):
                argument.annotation = ResolvedType(
                    QualifiedName.from_str("typing.Optional"), [argument.annotation]
                )

    @staticmethod
    def _annotation_variants(argument):
        annotation = argument.annotation
        if isinstance(
            annotation, ResolvedType
        ) and annotation.name == QualifiedName.from_str("typing.Union"):
            return frozenset(str(p) for p in annotation.parameters or [])
        return frozenset({str(annotation)})

    @classmethod
    def _covers(cls, f: Function, g: Function) -> bool:
        # f covers g if every call matching g also matches f with the same return type
        if str(f.returns) != str(g.returns) or len(f.args) != len(g.args):
            return False
        return all(
            f_arg.name == g_arg.name
            and str(f_arg.default) == str(g_arg.default)
            and cls._annotation_variants(f_arg) >= cls._annotation_variants(g_arg)
            for f_arg, g_arg in zip(f.args, g.args)
        )

    def _drop_redundant_overloads(self, class_):
        # Widening a parameter type can make a sibling overload redundant; drop it
        by_name = defaultdict(list)
        for method in class_.methods:
            by_name[method.function.name].append(method)
        removed = set()
        for methods in by_name.values():
            for g in methods:
                if any(
                    f is not g
                    and id(f) not in removed
                    and f.modifier == g.modifier
                    and self._covers(f.function, g.function)
                    for f in methods
                ):
                    removed.add(id(g))
        class_.methods = [m for m in class_.methods if id(m) not in removed]
        remaining = defaultdict(list)
        for method in class_.methods:
            remaining[method.function.name].append(method)
        for methods in remaining.values():
            if len(methods) == 1:
                methods[0].function.decorators = [
                    d
                    for d in methods[0].function.decorators
                    if str(d) != "typing.overload"
                ]

    def write_module(
        self, module: Module, printer: Printer, to: Path, sub_dir: Optional[Path] = None
    ):
        # The bindings name a class "Exception", shadowing the builtin in the stub. Refer
        # to the base classes as builtins.Exception to keep the names resolvable.
        exception_name = QualifiedName.from_str("Exception")
        builtin_exception_name = QualifiedName.from_str("builtins.Exception")
        for cls in module.classes:
            if exception_name in cls.bases:
                cls.bases = [
                    builtin_exception_name if base == exception_name else base
                    for base in cls.bases
                ]
                module.imports.add(
                    Import(name=None, origin=QualifiedName.from_str("builtins"))
                )
        for function in module.functions:
            self._patch_function(function)
        for cls in module.classes:
            for method in cls.methods:
                self._patch_function(method.function)
            for prop in cls.properties:
                if prop.setter is not None:
                    self._patch_function(prop.setter)
            self._drop_redundant_overloads(cls)
            for field in cls.fields:
                if (
                    field.attribute.annotation is not None
                    and field.attribute.annotation.name in self.alternative_types
                ):
                    converted_types = [
                        ResolvedType(e)
                        for e in self.alternative_types[field.attribute.annotation.name]
                    ]
                    field.attribute.annotation = ResolvedType(
                        QualifiedName.from_str("typing.Union"),
                        [field.attribute.annotation] + converted_types,
                    )
        super().write_module(module, printer, to, sub_dir=sub_dir)


class CustomPrinter(Printer):
    # Condition and Measure intentionally return Condition from __eq__/__ne__ to build
    # reaction expressions; suppress the resulting mypy complaints in the stubs.
    def print_function(self, func: Function) -> list:
        result = super().print_function(func)
        if func.name in ("__eq__", "__ne__") and str(func.returns) != "bool":
            # mypy anchors the override error to the first line of the (decorated) def
            result[0] += "  # type: ignore[override]"
        return result

    def print_attribute(self, attr) -> list:
        result = super().print_attribute(attr)
        if (
            attr.name == "__hash__"
            and attr.value is not None
            and str(attr.value) == "None"
        ):
            result[0] += "  # type: ignore[assignment]"
        return result


IMPLICIT_CONVERSIONS = [
    ("bool", "Condition"),
    ("float", "RelativeDynamicsFactor"),
    ("Affine", "RobotPose"),
    ("Twist", "RobotVelocity"),
    ("RobotPose", "CartesianState"),
    ("Affine", "CartesianState"),
    ("list[float]", "JointState"),
    ("numpy.ndarray", "JointState"),
]

alternatives = defaultdict(list)
for from_type, to_type in IMPLICIT_CONVERSIONS:
    alternatives[to_type].append(from_type)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - [%(levelname)7s] %(message)s",
    )
    args = arg_parser().parse_args()

    parser = stub_parser_from_args(args)

    printer = CustomPrinter(
        invalid_expr_as_ellipses=not args.print_invalid_expressions_as_is
    )

    out_dir, sub_dir = to_output_and_subdir(
        output_dir=args.output_dir,
        module_name=args.module_name,
        root_suffix=args.root_suffix,
    )

    run(
        parser,
        printer,
        args.module_name,
        out_dir,
        sub_dir=sub_dir,
        dry_run=args.dry_run,
        writer=CustomWriter(alternatives, stub_ext=args.stub_extension),
    )
