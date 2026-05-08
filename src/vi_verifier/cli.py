from __future__ import annotations

import argparse
import json
import sys

from .eval_cases import get_eval_cases, run_eval_suite
from .paper_compat import get_paper_compat_rows, render_paper_matrix_text
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vi-verify")
    sub = parser.add_subparsers(dest="command", required=True)

    run_case = sub.add_parser("run-case")
    run_case.add_argument("--case", required=True, dest="case_name")

    sub.add_parser("run-suite")
    sub.add_parser("list-cases")
    paper_matrix = sub.add_parser("paper-matrix")
    paper_matrix.add_argument("--format", choices=["text", "json"], default="text", dest="out_format")

    args = parser.parse_args(argv)
    if args.command == "run-case":
        return _run_case(args.case_name)
    if args.command == "run-suite":
        return _run_suite()
    if args.command == "list-cases":
        return _list_cases()
    if args.command == "paper-matrix":
        return _paper_matrix(args.out_format)
    return 2


if __name__ == "__main__":
    sys.exit(main())
