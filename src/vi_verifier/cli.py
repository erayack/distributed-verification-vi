from __future__ import annotations

import argparse
import json
import sys

from .eval_cases import get_eval_cases, run_eval_suite
from .paper_compat import get_paper_compat_rows, render_paper_matrix_text
from .proof_reproduction import (
    LowerBoundInput,
    compute_deterministic_lb,
    run_reduction_reproduction_suite,
)
from .verifier import Verifier


def _run_case(case_name: str) -> int:
    cases = {c.name: c for c in get_eval_cases()}
    if case_name not in cases:
        print(f"unknown case: {case_name}")
        return 2
    case = cases[case_name]
    result = Verifier().verify(case.graph_input, case.task)
    ok = result.verdict == case.expected
    status = "PASS" if ok else "FAIL"
    print(f"{status} {case.name}: expected={case.expected} got={result.verdict}")
    return 0 if ok else 1


def _run_suite() -> int:
    rows = run_eval_suite()
    passed = 0
    for case, result in rows:
        ok = result.verdict == case.expected
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"{status} {case.name}: expected={case.expected} got={result.verdict}")
    total = len(rows)
    print(f"Summary: {passed}/{total} passed")
    return 0 if passed == total else 1


def _list_cases() -> int:
    for case in get_eval_cases():
        print(case.name)
    return 0


def _paper_matrix(out_format: str) -> int:
    rows = get_paper_compat_rows()
    if out_format == "json":
        payload = [
            {
                "predicate": row.predicate,
                "paper_section": row.paper_section,
                "paper_rule": row.paper_rule,
                "status": row.status,
                "implementation_hook": row.implementation_hook,
                "notes": row.notes,
            }
            for row in rows
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(render_paper_matrix_text(rows))
    return 0


def _reproduce_lower_bounds(
    out_format: str, p: int | None, bandwidth: int | None, n: int | None
) -> int:
    sample_inputs = (
        [LowerBoundInput(p=p, B=bandwidth, n=n)]
        if p is not None and bandwidth is not None and n is not None
        else [
            LowerBoundInput(p=1, B=1, n=8),
            LowerBoundInput(p=2, B=3, n=(2**5) * 2 * 3),
            LowerBoundInput(p=3, B=2, n=(3**7) * 3 * 2),
        ]
    )
    lb_rows = []
    for lb_input in sample_inputs:
        result = compute_deterministic_lb(lb_input)
        lb_rows.append(
            {
                "p": lb_input.p,
                "B": lb_input.B,
                "n": lb_input.n,
                "valid_domain": result.valid_domain,
                "witness_base": result.witness_base,
                "exponent": result.exponent,
                "lb_value": result.lb_value,
            }
        )
    reductions = run_reduction_reproduction_suite()
    reductions_payload = [
        {
            "reduction_name": row.reduction_name,
            "passed": row.passed,
            "counterexample_case": row.counterexample_case,
        }
        for row in reductions
    ]

    if out_format == "json":
        print(
            json.dumps(
                {
                    "deterministic_lb": lb_rows,
                    "reduction_checks": reductions_payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("deterministic_lb")
        for row in lb_rows:
            print(
                f"p={row['p']} B={row['B']} n={row['n']} "
                f"valid_domain={row['valid_domain']} witness_base={row['witness_base']} "
                f"exponent={row['exponent']:.8f} lb_value={row['lb_value']:.8f}"
            )
        print("reduction_checks")
        for row in reductions_payload:
            status = "PASS" if row["passed"] else "FAIL"
            print(f"{status} {row['reduction_name']} counterexample={row['counterexample_case']}")

    return 0 if all(row["passed"] for row in reductions_payload) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vi-verify")
    sub = parser.add_subparsers(dest="command", required=True)

    run_case = sub.add_parser("run-case")
    run_case.add_argument("--case", required=True, dest="case_name")

    sub.add_parser("run-suite")
    sub.add_parser("list-cases")
    paper_matrix = sub.add_parser("paper-matrix")
    paper_matrix.add_argument(
        "--format", choices=["text", "json"], default="text", dest="out_format"
    )
    reproduce = sub.add_parser("reproduce-lower-bounds")
    reproduce.add_argument("--format", choices=["text", "json"], default="text", dest="out_format")
    reproduce.add_argument("--p", type=int, default=None)
    reproduce.add_argument("--B", type=int, default=None, dest="bandwidth")
    reproduce.add_argument("--n", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "run-case":
        return _run_case(args.case_name)
    if args.command == "run-suite":
        return _run_suite()
    if args.command == "list-cases":
        return _list_cases()
    if args.command == "paper-matrix":
        return _paper_matrix(args.out_format)
    if args.command == "reproduce-lower-bounds":
        if any(value is None for value in (args.p, args.bandwidth, args.n)) and any(
            value is not None for value in (args.p, args.bandwidth, args.n)
        ):
            print("must provide all of --p, --B and --n together")
            return 2
        return _reproduce_lower_bounds(args.out_format, args.p, args.bandwidth, args.n)
    return 2


if __name__ == "__main__":
    sys.exit(main())
