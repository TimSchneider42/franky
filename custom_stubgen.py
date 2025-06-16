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
from pybind11_stubgen.structs import Function, ResolvedType, Module


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

    def write_module(
        self, module: Module, printer: Printer, to: Path, sub_dir: Optional[Path] = None
    ):
        for cls in module.classes:
            for method in cls.methods:
                self._patch_function(method.function)
            for prop in cls.properties:
                if prop.setter is not None:
                    self._patch_function(prop.setter)
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


IMPLICIT_CONVERSIONS = [
    ("bool", "Condition"),
    ("float", "RelativeDynamicsFactor"),
    ("Affine", "RobotPose"),
    ("Twist", "RobotVelocity"),
    ("RobotPose", "CartesianState"),
    ("Affine", "CartesianState"),
    ("list[float]", "JointState"),
    ("np.ndarray", "JointState"),
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

    printer = Printer(invalid_expr_as_ellipses=not args.print_invalid_expressions_as_is)

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
