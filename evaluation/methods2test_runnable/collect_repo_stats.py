import argparse
import os
import json
import shutil
from pathlib import Path
from collections import Counter
from git import Repo, cmd, exc
from tqdm import tqdm
from datasets import load_dataset
import multiprocessing
import lizard

SCRIPT_PATH: Path = Path(os.path.abspath(__file__))
SCRIPT_DIR: Path = SCRIPT_PATH.parent

os.environ["GIT_TERMINAL_PROMPT"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="Collect repository-level statistics for methods2test repos.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to store results")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--tmp_dir", type=str, default="tmp", help="Temporary directory for caching and repos")
    return parser.parse_args()


def dir_size_bytes(path):
    """Total size in bytes, excluding .git directory."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        # Skip .git directory
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp) and not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def count_files(path):
    """Total number of files, excluding .git directory."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        count += len(filenames)
    return count


def collect_java_files(path):
    """Collect all .java file paths, excluding .git directory."""
    java_files = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for f in filenames:
            if f.endswith(".java"):
                java_files.append(os.path.join(dirpath, f))
    return java_files


def is_test_file(filepath):
    """Heuristic: a Java file is a test class if it lives under a test source directory
    or its filename matches common test naming conventions."""
    parts = Path(filepath).parts
    # Check for standard test source roots
    for i, part in enumerate(parts):
        if part in ("test", "tests", "test-src"):
            return True
        if part == "src" and i + 1 < len(parts) and parts[i + 1] == "test":
            return True
    name = Path(filepath).stem
    if name.startswith("Test") or name.endswith("Test") or name.endswith("Tests") or name.endswith("TestCase"):
        return True
    return False


def count_java_classes(java_files):
    """Count production and test classes from Java file list."""
    prod = 0
    test = 0
    for fp in java_files:
        if is_test_file(fp):
            test += 1
        else:
            prod += 1
    return prod, test


def detect_build_system(repo_path):
    """Detect the build system used by the repository."""
    indicators = {
        "Maven": ["pom.xml"],
        "Gradle": ["build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"],
        "Ant": ["build.xml"],
        "sbt": ["build.sbt"],
        "Bazel": ["BUILD", "BUILD.bazel", "WORKSPACE", "WORKSPACE.bazel"],
    }
    detected = []
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for build_sys, markers in indicators.items():
            for marker in markers:
                if marker in filenames and build_sys not in detected:
                    detected.append(build_sys)
    if not detected:
        return "Other"
    return ",".join(detected)


def count_modules(repo_path):
    """Count number of modules (sub-pom.xml / sub-build.gradle directories)."""
    module_count = 0
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        if "pom.xml" in filenames or "build.gradle" in filenames or "build.gradle.kts" in filenames:
            module_count += 1
    return module_count


def analyze_with_lizard(java_files):
    """Run lizard on Java files to get LOC and CCN metrics."""
    total_nloc = 0
    total_ccn = 0
    total_functions = 0
    for fp in java_files:
        try:
            analysis = lizard.analyze_file(fp)
            total_nloc += analysis.nloc
            for func in analysis.function_list:
                total_ccn += func.cyclomatic_complexity
                total_functions += 1
        except Exception:
            continue
    avg_ccn = total_ccn / total_functions if total_functions > 0 else 0.0
    return total_nloc, total_ccn, total_functions, avg_ccn


def process_repo(task):
    """Process a single repository: shallow clone at golden commit and collect stats."""
    repo_id, repo_url, commit_hash, tmp_dir = task

    repo_name = repo_url.rstrip("/").split("/")[-1]
    local_dir = tmp_dir / "repos" / str(os.getpid())
    os.makedirs(local_dir, exist_ok=True)
    repo_path = local_dir / repo_name

    result = {"repo_id": repo_id, "repo_url": repo_url, "commit": commit_hash}

    try:
        # Shallow clone at specific commit
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path, ignore_errors=True)

        repo = Repo.init(repo_path)
        repo.create_remote("origin", repo_url)
        repo.git.fetch("--depth", "1", "origin", commit_hash)
        repo.git.checkout("FETCH_HEAD")

        # Collect stats
        result["size_bytes"] = dir_size_bytes(repo_path)
        result["total_files"] = count_files(repo_path)

        java_files = collect_java_files(repo_path)
        prod_classes, test_classes = count_java_classes(java_files)
        result["java_classes_production"] = prod_classes
        result["java_classes_test"] = test_classes
        result["java_classes_total"] = prod_classes + test_classes

        result["modules"] = count_modules(repo_path)
        result["build_system"] = detect_build_system(repo_path)

        total_nloc, total_ccn, total_functions, avg_ccn = analyze_with_lizard(java_files)
        result["java_loc"] = total_nloc
        result["total_ccn"] = total_ccn
        result["total_functions"] = total_functions
        result["avg_ccn"] = round(avg_ccn, 4)

        result["status"] = "success"

    except exc.GitCommandError as e:
        result["status"] = "git_error"
        result["error"] = str(e)[:500]
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
    finally:
        try:
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True)
            if os.path.exists(local_dir) and not os.listdir(local_dir):
                shutil.rmtree(local_dir, ignore_errors=True)
        except Exception:
            pass

    return result


def main(args):
    tmp_dir = SCRIPT_DIR / args.tmp_dir
    save_path = SCRIPT_DIR / Path(args.output_dir) / f"repo_stats_{args.split}.jsonl"

    os.makedirs(save_path.parent, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    raw_ds = load_dataset("andstor/methods2test_raw", split=args.split, cache_dir=tmp_dir / "cache")
    meta_ds = load_dataset("andstor/methods2test_meta", "golden_commit", split=args.split, cache_dir=tmp_dir / "cache")

    # Build mapping: id -> (repo_url, commit)
    meta_map = {row["id"]: row["commit"] for row in meta_ds}

    # Group by repo_id and pick the most common golden commit per repo
    repo_info = {}  # repo_id -> {"url": ..., "commits": [...]}
    for example in raw_ds:
        example_id = example["id"]
        repo_id = example_id.split("_")[0]
        repo_url = example["repository"]["url"]
        commit = meta_map.get(example_id)
        if commit is None:
            continue
        if repo_id not in repo_info:
            repo_info[repo_id] = {"url": repo_url, "commits": []}
        repo_info[repo_id]["commits"].append(commit)

    # Select most common commit per repo
    repo_tasks = {}
    for repo_id, info in repo_info.items():
        most_common_commit = Counter(info["commits"]).most_common(1)[0][0]
        repo_tasks[repo_id] = (info["url"], most_common_commit)

    print(f"Total unique repos: {len(repo_tasks)}")

    # Resume: collect already processed repo_ids
    processed_ids = set()
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj and "repo_id" in obj:
                        processed_ids.add(obj["repo_id"])
                except Exception:
                    continue

    # Prepare tasks, skipping already processed repos
    tasks = []
    for repo_id, (url, commit) in repo_tasks.items():
        if repo_id not in processed_ids:
            tasks.append((repo_id, url, commit, tmp_dir))

    print(f"Repos to process: {len(tasks)} (skipping {len(processed_ids)} already processed)")

    if not tasks:
        print("All repos already processed.")
        return

    with tqdm(total=len(tasks), desc="Processing repos", unit="repo") as pbar:
        with open(save_path, "a") as output_file:
            with multiprocessing.Pool(processes=args.num_proc) as pool:
                for result in pool.imap_unordered(process_repo, tasks):
                    output_file.write(json.dumps(result) + "\n")
                    output_file.flush()
                    pbar.update(1)

    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
