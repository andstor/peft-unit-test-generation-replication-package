"""Evaluate generated Defects4J fixed-method predictions.

For each generated record (id, prompt, reference, prediction):
  1. Look up the (project_id, bug_id, class_file, method) via the
     `andstor/defects4j_fixed_runnable` HuggingFace dataset.
  2. Check out the fixed version of the bug with `defects4j checkout`.
  3. Replace the original method body in the source file with the prediction.
  4. Run `defects4j compile` and `defects4j test`.
  5. Record an execution status in the same format as humaneval-x and
     methods2test_runnable:
        {"id": <int>, "status": "<status>"}

Status semantics follow the paper (§4.9) and the surefire-based evaluators:
  success           - all executed tests passed
  failed            - at least one test failed an assertion
  error             - at least one test threw an unexpected runtime exception
  skipped           - every executed test was skipped (no failures/errors)
  compilation error - `defects4j compile` returned non-zero
  timeout           - compile or test phase exceeded the configured timeout
  exception         - environmental issue (checkout failed, missing class file,
                      anchor mismatch, parse failure, unhandled Python error)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

# How many trailing chars of compile output to embed in the results file
# when a record fails to compile. Set to 0 to disable.
COMPILE_ERROR_TAIL = 2000

logger = logging.getLogger(__name__)

SCRIPT_DIR: Path = Path(os.path.abspath(__file__)).parent

# Per-step subprocess timeouts (seconds). Set to None to disable.
COMPILE_TIMEOUT: int | None = None
TEST_TIMEOUT: int | None = None
DEBUG: bool = False


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated Defects4J fixed-method predictions."
    )
    parser.add_argument("--data_dir", type=str,
                        default="../../data/defects4j_fixed_runnable/fixed",
                        help="Directory containing generated jsonl files.")
    parser.add_argument("--output_dir", type=str,
                        default="../../data/defects4j_fixed_runnable/executed",
                        help="Directory where results.jsonl files are written.")
    parser.add_argument("--tmp_dir", type=str, default="tmp",
                        help="Temporary directory for checkouts and HF cache.")
    parser.add_argument("--num_proc", type=int, default=1,
                        help="Number of worker processes (one checkout dir per worker).")
    parser.add_argument("--dataset", type=str,
                        default="andstor/defects4j_fixed",
                        help="HuggingFace dataset providing id -> bug metadata. "
                             "Must expose project_id, bug_id, class.file, method.body, tests.")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Stream subprocess output to terminal.")
    parser.add_argument("--compile_timeout", type=int, default=None,
                        help="Per-prediction compile timeout in seconds (default: no limit).")
    parser.add_argument("--test_timeout", type=int, default=300,
                        help="Per-prediction test timeout in seconds (default: 300).")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def find_generation_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.rglob("*.jsonl") if p.is_file())


def output_path_for(input_file: Path, data_dir: Path, output_dir: Path) -> Path:
    """Mirror the relative directory structure (peft/namespace/model)."""
    rel = input_file.parent.relative_to(data_dir)
    return output_dir / rel / "results.jsonl"


def trim_trailing_class_brace(text: str) -> str:
    """Strip the last `}` (the class-level closing brace appended in source/target)."""
    idx = text.rfind("}")
    return text[:idx] if idx != -1 else text


def load_completed_ids(results_file: Path) -> set[int]:
    if not results_file.exists():
        return set()
    done: set[int] = set()
    with results_file.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(int(obj["id"]))
            except Exception:
                continue
    return done


def append_result(results_file: Path, record: dict) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
# Defects4J wrappers
# --------------------------------------------------------------------------- #
_WORKER_ID: int = 0  # set per-worker in _process_file / _worker_run


def _run_debug(cmd: list[str], *, check: bool = False,
               timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run *cmd* while streaming every output line prefixed with worker info.

    If *timeout* is set, a watchdog thread kills the entire process group
    after that many seconds — even when the child produces no output.
    """
    import signal
    import threading

    prefix = f"\033[90m[worker-{_WORKER_ID}]\033[0m "
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group so we can kill the tree
    )
    timed_out = False

    def _watchdog():
        nonlocal timed_out
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()

    timer = None
    if timeout is not None:
        timer = threading.Timer(timeout, _watchdog)
        timer.start()

    lines: list[str] = []
    assert proc.stdout is not None
    try:
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace")
            lines.append(line)
            print(f"{prefix}{line}", end="", flush=True)
    finally:
        proc.wait()
        if timer is not None:
            timer.cancel()

    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    stdout_bytes = "".join(lines).encode("utf-8")
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout=stdout_bytes)


def defects4j_checkout(project: str, bug_id: str, workdir: Path) -> None:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.parent.mkdir(parents=True, exist_ok=True)
    if DEBUG:
        _run_debug(
            ["defects4j", "checkout",
             "-p", project, "-v", f"{bug_id}f", "-w", str(workdir)],
            check=True,
        )
    else:
        subprocess.run(
            ["defects4j", "checkout",
             "-p", project, "-v", f"{bug_id}f", "-w", str(workdir)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )


def defects4j_compile(workdir: Path) -> tuple[int, str]:
    if DEBUG:
        proc = _run_debug(
            ["defects4j", "compile", "-w", str(workdir)],
            timeout=COMPILE_TIMEOUT,
        )
        return proc.returncode, proc.stdout.decode("utf-8", errors="replace")
    proc = subprocess.run(
        ["defects4j", "compile", "-w", str(workdir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=COMPILE_TIMEOUT,
    )
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace")


_FAILING_RE = re.compile(r"^Failing tests:\s*(\d+)\s*$", re.MULTILINE)

# Where Ant / Maven / Gradle drop JUnit XML reports inside a defects4j checkout.
_REPORT_DIR_GLOBS = (
    "**/surefire-reports",
    "**/test-output",
    "**/test-results",
)
_REPORT_FILE_GLOB = "TEST-*.xml"


def _clean_test_reports(workdir: Path) -> None:
    """Delete stale JUnit report directories so we only see the next run's output."""
    for pattern in _REPORT_DIR_GLOBS:
        for d in workdir.glob(pattern):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)


def _find_junit_reports(workdir: Path) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for dir_pattern in _REPORT_DIR_GLOBS:
        for d in workdir.glob(dir_pattern):
            if not d.is_dir():
                continue
            for xml in d.rglob(_REPORT_FILE_GLOB):
                if xml not in seen:
                    seen.add(xml)
                    out.append(xml)
    return out


def _aggregate_junit(reports: list[Path]) -> dict | None:
    if not reports:
        return None
    agg = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    parsed_any = False
    for path in reports:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        for s in suites:
            parsed_any = True
            for k in agg:
                try:
                    agg[k] += int(s.attrib.get(k, 0))
                except ValueError:
                    pass
    return agg if parsed_any else None


def defects4j_test(
    workdir: Path, tests: list[str] | None = None
) -> tuple[int, dict | None, int, str]:
    """Run `defects4j test` and collect both stdout and JUnit XML results.

    If `tests` is provided, run each one individually with `defects4j test -t <name>`
    (defects4j accepts only one `-t` per invocation) and aggregate the XML reports.
    Otherwise run the project's default trigger + relevant tests.

    Returns (last_returncode, junit_counts_or_None, num_failing_from_stdout,
    raw_stdout). `junit_counts_or_None` aggregates {tests, failures, errors,
    skipped} across all invocations.
    """
    _clean_test_reports(workdir)
    invocations = [("-t", t) for t in tests] if tests else [None]

    last_rc = 0
    failing_total = 0
    saw_failing_line = False
    combined_out: list[str] = []

    for inv in invocations:
        cmd = ["defects4j", "test", "-w", str(workdir)]
        if inv is not None:
            cmd += list(inv)
        if DEBUG:
            proc = _run_debug(cmd, timeout=TEST_TIMEOUT)
            out = proc.stdout.decode("utf-8", errors="replace")
        else:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=TEST_TIMEOUT,
                start_new_session=True,
            )
            out = proc.stdout.decode("utf-8", errors="replace")
        combined_out.append(out)
        last_rc = proc.returncode
        m = _FAILING_RE.search(out)
        if m:
            saw_failing_line = True
            failing_total += int(m.group(1))

    counts = _aggregate_junit(_find_junit_reports(workdir))
    failing = failing_total if saw_failing_line else -1
    return last_rc, counts, failing, "\n".join(combined_out)


def classify_test_result(counts: dict | None, failing_from_stdout: int) -> str:
    """Map a `defects4j test` outcome onto the paper's status taxonomy."""
    if counts is not None and counts["tests"] > 0:
        if counts["errors"] > 0:
            return "error"
        if counts["failures"] > 0:
            return "failed"
        if counts["skipped"] >= counts["tests"]:
            return "skipped"
        return "success"
    # Fall back to stdout's `Failing tests: N` (cannot distinguish error vs failed).
    if failing_from_stdout < 0:
        return "exception"
    return "success" if failing_from_stdout == 0 else "failed"


# --------------------------------------------------------------------------- #
# Method replacement
# --------------------------------------------------------------------------- #
def replace_method_body(
    source_file: Path, original_target: str, new_prediction: str
) -> tuple[bool, str]:
    """Replace the original method body (anchored by `original_target`) with the
    prediction.

    `original_target` is expected to be the method body starting from just
    after the opening `{` and ending with the method's closing `}` (the form
    produced by `build_meta_lookup`). The prediction may end with the
    surrounding class-level `}`; we strip that so we don't duplicate braces.

    Returns (ok, reason). On failure `reason` is one of "missing", "ambiguous".
    Line endings in the anchor and source file are normalized to LF before
    matching to avoid CRLF/LF mismatches between the dataset and the checkout.
    """
    anchor = original_target.replace("\r\n", "\n").replace("\r", "\n").rstrip()
    replacement = trim_trailing_class_brace(new_prediction).replace(
        "\r\n", "\n").replace("\r", "\n").rstrip()

    raw = source_file.read_text(encoding="utf-8", errors="replace")
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    count = text.count(anchor)
    if count == 0:
        return False, "missing"
    if count > 1:
        return False, "ambiguous"

    new_text = text.replace(anchor, replacement, 1)
    source_file.write_text(new_text, encoding="utf-8")
    return True, "ok"


# --------------------------------------------------------------------------- #
# Per-bug evaluation
# --------------------------------------------------------------------------- #
def evaluate_bug_group(
    project_id: str,
    bug_id: str,
    rows: list[dict],
    meta_lookup: dict[int, dict],
    results_file: Path,
    tmp_dir: Path,
    worker_id: int,
    progress: "tqdm | None" = None,
) -> None:
    """Evaluate every prediction belonging to a single (project, bug)."""
    workdir = tmp_dir / "checkouts" / f"worker-{worker_id}" / f"{project_id}-{bug_id}"

    logger.info("[worker-%s] checkout %s-%s", worker_id, project_id, bug_id)
    try:
        defects4j_checkout(project_id, bug_id, workdir)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
        logger.error("[worker-%s] checkout failed for %s-%s: %s", worker_id, project_id, bug_id, err)
        for row in rows:
            append_result(results_file, {"id": int(row["id"]), "status": "exception"})
            if progress is not None:
                progress.update(1)
        return

    for row in rows:
        rec_id = int(row["id"])
        meta = meta_lookup[rec_id]
        class_file = workdir / meta["class_file"]

        if not class_file.exists():
            logger.warning("Missing class file %s for id %s", class_file, rec_id)
            append_result(results_file, {"id": rec_id, "status": "exception",
                                          "error": f"missing class file: {meta['class_file']}"})
            if progress is not None:
                progress.update(1)
            continue

        original_text = class_file.read_text(encoding="utf-8", errors="replace")
        try:
            ok, reason = replace_method_body(
                class_file,
                original_target=meta["anchor"],
                new_prediction=row["prediction"],
            )
            if not ok:
                logger.info("[worker-%s] id %s: anchor %s", worker_id, rec_id, reason)
                append_result(results_file, {"id": rec_id, "status": "exception",
                                              "error": f"anchor {reason} in source file"})
                continue

            try:
                rc, compile_out = defects4j_compile(workdir)
            except subprocess.TimeoutExpired:
                append_result(results_file, {"id": rec_id, "status": "timeout",
                                              "error": "compile timeout"})
                continue

            if rc != 0:
                record = {"id": rec_id, "status": "compilation error"}
                if COMPILE_ERROR_TAIL > 0:
                    record["compile_error"] = compile_out[-COMPILE_ERROR_TAIL:]
                append_result(results_file, record)
                continue

            try:
                _, counts, failing, _ = defects4j_test(workdir, tests=meta.get("tests"))
            except subprocess.TimeoutExpired:
                append_result(results_file, {"id": rec_id, "status": "timeout",
                                              "error": "test timeout"})
                continue

            status = classify_test_result(counts, failing)
            record: dict = {"id": rec_id, "status": status}
            logger.info("[worker-%s] id %s: %s", worker_id, rec_id, status)
            if status == "exception":
                record["error"] = "could not parse 'Failing tests:' from defects4j test output"
            append_result(results_file, record)

        except Exception as e:  # noqa: BLE001
            logger.exception("[worker-%s] Unhandled error for id %s: %s", worker_id, rec_id, e)
            append_result(results_file, {"id": rec_id, "status": "exception", "error": str(e)})
        finally:
            # Always restore the original source before the next prediction.
            class_file.write_text(original_text, encoding="utf-8")
            if progress is not None:
                progress.update(1)

    # Clean up checkout to bound disk usage.
    shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Worker dispatch
# --------------------------------------------------------------------------- #
def _process_file(task: tuple) -> None:
    (file_path, output_file, meta_lookup, tmp_dir, worker_id) = task
    global _WORKER_ID  # noqa: PLW0603
    _WORKER_ID = worker_id

    completed = load_completed_ids(output_file)
    df = pd.read_json(file_path, orient="records", lines=True, dtype=False)
    if df.empty:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.touch(exist_ok=True)
        return

    rows = []
    for _, r in df.iterrows():
        rid = int(r["id"])
        if rid in completed:
            continue
        if rid not in meta_lookup:
            append_result(output_file, {"id": rid, "status": "exception"})
            continue
        rows.append({"id": rid, "prediction": r["prediction"]})

    # Group by (project, bug) so each checkout is reused.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        meta = meta_lookup[r["id"]]
        groups[(meta["project_id"], meta["bug_id"])].append(r)

    progress = tqdm(
        total=len(rows),
        desc=str(file_path.relative_to(file_path.parents[3]))
        if len(file_path.parents) >= 4 else file_path.name,
        unit="pred",
        dynamic_ncols=True,
        leave=False,
        position=worker_id % 8 if worker_id else 0,
    )
    try:
        for (project_id, bug_id), group_rows in groups.items():
            progress.set_postfix_str(f"{project_id}-{bug_id}")
            evaluate_bug_group(
                project_id=project_id,
                bug_id=bug_id,
                rows=group_rows,
                meta_lookup=meta_lookup,
                results_file=output_file,
                tmp_dir=tmp_dir,
                worker_id=worker_id,
                progress=progress,
            )
    finally:
        progress.close()


def _worker_init(meta_lookup_arg: dict, tmp_dir_arg: Path, debug_arg: bool = False,
                 compile_timeout_arg: int | None = None,
                 test_timeout_arg: int | None = 300):
    global _META_LOOKUP, _TMP_DIR, DEBUG, COMPILE_TIMEOUT, TEST_TIMEOUT  # noqa: PLW0603
    _META_LOOKUP = meta_lookup_arg
    _TMP_DIR = tmp_dir_arg
    DEBUG = debug_arg
    COMPILE_TIMEOUT = compile_timeout_arg
    TEST_TIMEOUT = test_timeout_arg


def _worker_run(args: tuple) -> None:
    file_path, output_file = args
    worker_id = os.getpid()
    _process_file((file_path, output_file, _META_LOOKUP, _TMP_DIR, worker_id))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_meta_lookup(dataset_name: str, split: str, cache_dir: Path) -> dict[int, dict]:
    """Build id -> bug metadata.

    Supports two dataset shapes:
      * `andstor/defects4j_fixed`         — has nested `class`, `method`, `tests`.
      * `andstor/defects4j_fixed_runnable` — has flat `class_file`, `method`,
        `source`, `target` (no per-method tests).

    The anchor used to locate the original method body in the checked-out
    source file is derived as the substring of the method body after its first
    `{` (i.e. body + closing brace), so the prediction — which the model emits
    starting from inside the method body — can be inserted in place.
    """
    ds = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir))
    out: dict[int, dict] = {}
    for row in ds:
        rid = int(row["id"])
        if isinstance(row.get("class"), dict):
            # andstor/defects4j_fixed schema
            class_file = row["class"]["file"]
            method_body: str = row["method"]["body"]
            brace_idx = method_body.find("{")
            anchor = method_body[brace_idx + 1:] if brace_idx != -1 else method_body
            tests = [t["name"] for t in (row.get("tests") or [])]
        else:
            # andstor/defects4j_fixed_runnable schema (no tests column)
            class_file = row["class_file"]
            anchor = trim_trailing_class_brace(row["target"]).rstrip()
            tests = []
        out[rid] = {
            "project_id": row["project_id"],
            "bug_id": row["bug_id"],
            "class_file": class_file,
            "anchor": anchor.rstrip(),
            "tests": tests,
        }
    return out


def main(args) -> None:
    data_dir = (SCRIPT_DIR / args.data_dir).resolve()
    output_dir = (SCRIPT_DIR / args.output_dir).resolve()
    tmp_dir = (SCRIPT_DIR / args.tmp_dir).resolve()
    cache_dir = tmp_dir / "cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading metadata dataset %s[%s]", args.dataset, args.split)
    meta_lookup = build_meta_lookup(args.dataset, args.split, cache_dir)
    logger.info("Loaded %d metadata rows", len(meta_lookup))

    files = find_generation_files(data_dir)
    logger.info("Found %d generation files under %s", len(files), data_dir)

    tasks = [(fp, output_path_for(fp, data_dir, output_dir)) for fp in files]

    if args.debug:
        global DEBUG  # noqa: PLW0603
        DEBUG = True

    global COMPILE_TIMEOUT, TEST_TIMEOUT  # noqa: PLW0603
    COMPILE_TIMEOUT = args.compile_timeout
    TEST_TIMEOUT = args.test_timeout

    if args.num_proc <= 1:
        for fp, out in tqdm(tasks, desc="Files", unit="file", dynamic_ncols=True):
            logger.info("Evaluating %s", fp)
            _process_file((fp, out, meta_lookup, tmp_dir, 0))
    else:
        with Pool(
            processes=args.num_proc,
            initializer=_worker_init,
            initargs=(meta_lookup, tmp_dir, DEBUG, COMPILE_TIMEOUT, TEST_TIMEOUT),
        ) as pool:
            for _ in tqdm(
                pool.imap_unordered(_worker_run, tasks),
                total=len(tasks),
                desc="Files",
                unit="file",
                dynamic_ncols=True,
            ):
                pass


class _TqdmHandler(logging.StreamHandler):
    """Logging handler that emits through tqdm.write() to avoid clobbering
    progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


class _ColorFormatter(logging.Formatter):
    """Logging formatter that adds ANSI colours per level."""

    _COLORS = {
        logging.DEBUG:    "\033[36m",     # cyan
        logging.INFO:     "\033[32m",     # green
        logging.WARNING:  "\033[33m",     # yellow
        logging.ERROR:    "\033[31m",     # red
        logging.CRITICAL: "\033[1;31m",   # bold red
    }
    _RESET = "\033[0m"

    def __init__(self, fmt: str, datefmt: str | None = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{self._RESET}"
        return super().format(record)


if __name__ == "__main__":
    _handler = _TqdmHandler()
    _handler.setFormatter(_ColorFormatter(
        fmt="%(asctime)s [%(levelname)s] [%(process)d] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.basicConfig(level=logging.INFO, handlers=[_handler])
    main(parse_args())
