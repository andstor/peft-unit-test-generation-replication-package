import argparse
import os
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from git import Repo, cmd, exc
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
import multiprocessing

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Find matching commits for test cases.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--local_dir", type=str, default=".tmp", help="Local directory for repositories")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to store results")
    parser.add_argument("--max_workers", type=int, default=5, help="Number of parallel threads")
    return parser.parse_args()

def get_file_existence_spans(repo, file_path):
    """Get spans of commits where the file exists."""
    try:
        log_output = repo.git.log('--diff-filter=AD', '--branches', '--tags', '--format=%H', '--', file_path)
        log_entries = log_output.splitlines()
        spans = []
        for i in range(0, len(log_entries), 2):
            start = log_entries[i]
            end = log_entries[i+1] if i+1 < len(log_entries) else repo.head.commit.hexsha
            spans.append((start, end))
        return spans
    except exc.GitCommandError:
        return []

def get_commits_between(repo, start_commit, end_commit):
    """Get list of commits between start and end."""
    try:
        return repo.git.rev_list(f"{start_commit}..{end_commit}").splitlines()
    except exc.GitCommandError:
        return []

def extract_test_contents(example):
    """Extract test method identifier and body."""
    return [
        example["test_class"]["identifier"],
        example["test_case"]["body"]
    ]

def extract_focal_contents(example):
    """Extract focal methods and body."""
    methods = [m["signature"] for m in example["focal_class"]["methods"]]
    return methods + [example["focal_method"]["body"]]

def commit_file_match(repo, commit_hash, file_path, strings):
    """Check if all strings exist in file at given commit."""
    try:
        repo.git.checkout(commit_hash, file_path)
        with open(os.path.join(repo.working_dir, file_path), "r") as file:
            file_contents = file.read()
            return all(string in file_contents for string in strings)
    except (exc.GitCommandError, FileNotFoundError, UnicodeDecodeError):
        return False

def get_candidates(repo, file_path, upper_datetime):
    """Get candidate commits where file exists and is within the date range."""
    spans = get_file_existence_spans(repo, file_path)
    candidates = set()
    for start, end in spans:
        commits = get_commits_between(repo, start, end)
        if commits:
            commits.append(end)
            for commit in commits:
                try:
                    commit_obj = repo.commit(commit)
                    if commit_obj.committed_datetime <= upper_datetime:
                        candidates.add(commit)
                except (exc.GitCommandError, ValueError):
                    continue
    return candidates

def process_group(task):
    """Process a group of test cases with the same mainnumber."""
    records, args, upper_datetime = task

    repo_url = records[0]["repository"]["url"]
    repo_name = repo_url.split('/')[-1]
    repo_path = os.path.join(args.local_dir, repo_name)
    
    try:
        # Clone repository
        if not os.path.exists(repo_path):
            cmd.Git().clone(repo_url, repo_path, kill_after_timeout=180)
        repo = Repo(repo_path)
        if repo.is_dirty():
            repo.git.reset('--hard')
        
        results = []
        for example in records:
            test_file = example["test_class"]["file"]
            focal_file = example["focal_class"]["file"]
            example_id = example["id"]
            
            test_candidates = get_candidates(repo, test_file, upper_datetime)
            focal_candidates = get_candidates(repo, focal_file, upper_datetime)
            candidates = test_candidates.intersection(focal_candidates)
            
            found = False
            for candidate in candidates:
                test_match = commit_file_match(repo, candidate, test_file, extract_test_contents(example))
                focal_match = commit_file_match(repo, candidate, focal_file, extract_focal_contents(example))
                if test_match and focal_match:
                    result = {
                        "id": example_id,
                        "commit": candidate,
                        "candidates": list(candidates)
                    }
                    found = True
                    break
            if not found:
                result = {
                    "id": example_id,
                    "commit": None,
                    "candidates": list(candidates)
                }
            results.append(result)
        return results
    except exc.GitCommandError:
        return [None]
    except BaseException as e:
        return [None]
    finally:
        # Cleanup: Remove cloned repository
        try:
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True)
        except BaseException:
            pass

def main():
    """Main function to process data8set using multiprocessing.Pool."""
    args = parse_args()
    
    # Initialize output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Load dataset
    ds = load_dataset("andstor/methods2test_raw")[args.split]

    # Group records by mainnumber
    grouped_records = {}
    for example in ds:
        mainnumber = example["id"].split("_")[0]
        grouped_records.setdefault(mainnumber, []).append(example)
    
    # Desired number of processes
    num_processes = args.max_workers
    
    # Prepare tasks
    tasks = []
    upper_datetime = datetime.strptime("2021-05-18T23:18:51.000Z", 
                                     "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    for mainnumber, records in grouped_records.items():
        tasks.append((records, args, upper_datetime))
    
    # Initialize progress bar
    with tqdm(total=len(tasks), desc="Processing", unit="repo") as progress_bar:
        # Initialize output file
        output_path = os.path.join(args.output_dir, f"commits_{args.split}.jsonl")
        with open(output_path, "w") as output_file:
            # Use multiprocessing.Pool to distribute tasks
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Use imap to lazily supply the tasks
                for results in pool.imap(process_group, tasks):
                    # Write results to output file
                    if results is None:
                        continue
                    for result in results:
                        output_file.write(json.dumps(result) + "\n")
                        output_file.flush()
                        # Update progress bar
                    progress_bar.update(1)

if __name__ == "__main__":
    main()