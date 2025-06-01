import shutil
import os
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from multiprocessing import Queue, Process, BoundedSemaphore
from src.test_executer import TestExecutor, TestCandidateDescriptor, FocalMethodDescriptor
import queue
import argparse

import logging
logger = logging.getLogger(__name__)


SCRIPT_PATH: Path = Path(os.path.abspath(__file__))
SCRIPT_DIR: Path = SCRIPT_PATH.parent

os.environ["GIT_TERMINAL_PROMPT"] = "0"  # Disable git terminal prompt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate buildable test repositories.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to store results. Relative to script directory.")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of parallel threads")
    parser.add_argument("--push_to_hub", action='store_true', help="Whether to push results to Hugging Face Hub")
    parser.add_argument("--tmp_dir", type=str, default=".tmp", help="Temporary directory for caching and repos. Relative to script directory.")

    return parser.parse_args()


def trim_end_brac(input_string):
        # Find the last occurrence of '}'
        brace_index = input_string.rfind('}')
        if brace_index != -1:
            # Return the substring up to and including '}'
            return input_string[:brace_index]
        return input_string  # Return the original string if '}' is not found


def constrained_iterator(sem: BoundedSemaphore, data: iter):
    for i in data:
        sem.acquire()
        yield i

def progress_tracker_worker(progressq: Queue, total: int):
    from tqdm import tqdm
    
    def smoothing_for_window(total_iterations, window_fraction=0.05):
        N = max(1, int(window_fraction * total_iterations))
        smoothing = 2 / (N + 1)
        return min(max(smoothing, 0), 1)

    pbar = tqdm(desc="Executing tests", total=total, smoothing=smoothing_for_window(total), dynamic_ncols=True)
    while True:
        try:
            item = progressq.get()
            if item is None:
                pbar.close()
                break
            pbar.update(item)
        except queue.Empty:
            continue   

# Worker to write data into different shard files
def jsonl_writer_worker(writeq: Queue, save_path: Path):
    while True:
        try:
            item = writeq.get()
            if item is None:
                break

            try:
                # Get data and its respective file from the queue
                data = item

                with open(save_path, "a") as jacoco_file:
                    jacoco_file.write(json.dumps(data) + "\n")
                    
            except Exception as e:
                print(f"Error writing data: {e}")
        except queue.Empty:
            continue  
    
def process_test(dataq: Queue, writeq: Queue, progressq: Queue, sem: BoundedSemaphore, save_path: Path, tmp_dir: Path):
    
    meta_ds = load_dataset("andstor/methods2test_meta", "golden_commit", split="test", cache_dir=tmp_dir / "cache")
    meta_ds_df = meta_ds.to_pandas().set_index("id")
    
    raw_ds = load_dataset("andstor/methods2test_raw", split="test", cache_dir=tmp_dir / "cache")
    raw_ds_df = raw_ds.to_pandas().set_index("id")
    
    
    while True:
        try:
            item = dataq.get()
            if item is None:
                logger.info("No more items to process, exiting worker")
                break
            test_id = item
        except queue.Empty:
            logger.info("No more items to process, exiting worker")
            continue
        data = {}
        data["id"] = None
        data["status"] = None
        data["build_tool"] = None
        
        try:
            from git import Repo
            data["id"] = test_id
            
            sample_raw = raw_ds_df.loc[test_id]
            sample_commit = meta_ds_df.loc[test_id]
            if sample_commit is None:
                logger.warning("No commit found for id:", test_id)
                sem.release()
                continue
            else:
                sample_commit = sample_commit["commit"]
            
            
            repo_url = sample_raw[ "repository"]["url"]
            repo_name = repo_url.split('/')[-1]

            local_dir = tmp_dir / "repos" / str(os.getpid())
            os.makedirs(local_dir, exist_ok=True)
            repo_path = local_dir / repo_name
            
            #check if the repo is already cloned
            if os.path.exists(repo_path):
                repo = Repo(repo_path)
                logger.info(f"Using existing repo at {repo_path}")
            else:
                logger.info(f"Cloning {repo_url} into {repo_path}")
                repo = Repo.init(repo_path)
                repo.create_remote("origin", repo_url)
                repo.git.fetch("--depth", "1", "origin", sample_commit)
                repo.git.checkout("FETCH_HEAD")
            
            repo.git.reset("--hard")
                
                
            # Main logic to execute the test case
            sample_raw = raw_ds_df.loc[test_id]

            testCandidate = TestCandidateDescriptor(
                function_identifier=sample_raw["test_case"]["identifier"],
                class_identifier=sample_raw["test_class"]["identifier"],
                file=sample_raw["test_class"]["file"]
            )
            
            focal_method = FocalMethodDescriptor(
                function_identifier=sample_raw["focal_method"]["identifier"],
                class_identifier=sample_raw["focal_class"]["identifier"],
                signature=sample_raw["focal_method"]["signature"],
                file=sample_raw["focal_class"]["file"],
            )
            
            executor = TestExecutor(repo)
            executor.register_unit_test(testCandidate)
            executor.register_focal_method(focal_method)
            executor.detect_build_tool()
            data["build_tool"] = executor.build_system.get_name()
            
            executor.install_coverage_tool()
            
            # First try to compile the original code
            #executor.clean()
            out, err, returncode = executor.execute()
            logger.debug(out)
            logger.debug(err)
            logger.debug(returncode)
            
            
            results = executor.get_results()
            status = None
            if results is None:
                logger.info("BUILD ERROR")
                status = "build error"
            elif results["skipped"] > 0:
                logger.info("TEST SKIPPED")
                status = "skipped"
            elif results["failures"] > 0:
                logger.info("TEST FAILED")
                status = "failed"
            elif results["errors"] > 0:
                logger.info("TEST ERROR")
                status = "error"
            else:
                status = "success"
                logger.info("TEST PASSED")
            
            
            coverage = None
            if results is not None:
                coverage = executor.get_coverage_report()
                logger.info(pd.Series(coverage).to_frame().T)

            data["status"] = status
            data["build_tool"] = executor.build_system.get_name()
            # add coverage to data without knowing the keys
            if coverage is not None:
                data.update(coverage)
            writeq.put((data))
            
            if status == "build error":
                logger.info("Build error in original code")
                continue
            if status == "skipped":
                logger.info("Test skipped in original code")
                continue
        
        except Exception as e:
            logger.error(e)
            data["status"] = "exception"

            with open(save_path, "a") as exception_file:
                exception_file.write(json.dumps(data) + "\n")

        finally:
            progressq.put(1)
            sem.release()
            # Signal progress
            try:
                if os.path.exists(repo_path):
                    shutil.rmtree(repo_path, ignore_errors=True)
                # Also cleanup the temp dir for this process id if empty
                if os.path.exists(local_dir) and not os.listdir(local_dir):
                    shutil.rmtree(local_dir, ignore_errors=True)
                
            except BaseException:
                pass


    
def main(args):
    
    num_proc = args.num_proc
    tmp_dir = SCRIPT_DIR / args.tmp_dir
    save_path = SCRIPT_DIR / Path(args.output_dir) / f"buildable_{args.split}.jsonl"
    
    small_ds = load_dataset("andstor/methods2test_small", "fm+fc+c+m+f+t+tc", split="test", cache_dir=tmp_dir / "cache")
    small_ds_df = small_ds.to_pandas()
    
    total_tests = len(small_ds_df)
    logger.info(f"Total number of test cases: {total_tests}")
    
    
    progressq = Queue()
    progress_tracker = Process(target=progress_tracker_worker, args=(progressq,total_tests))
    progress_tracker.start()
    
    
    
    # Number of shards (write processes)
    writeq = Queue()
    jsonl_writer = Process(target=jsonl_writer_worker, args=(writeq, save_path))
    jsonl_writer.start()
    
    
    sem = BoundedSemaphore(num_proc)
    dataq = Queue()
    process_test_workers = [
        Process(target=process_test, args=(dataq, writeq, progressq, sem, save_path, tmp_dir), daemon=True)
        for _ in range(num_proc)
    ]
    for p in process_test_workers:
        p.start()
    
    
    os.makedirs(save_path.parent, exist_ok=True)

    
    # Resume logic: collect already processed ids
    processed_ids = set()
    if os.path.exists(save_path):
        out_df = pd.read_json(save_path, orient='records', lines=True, dtype=False)
        if not out_df.empty:
            out_df = out_df.set_index("id")
            processed_ids.update(set(out_df.index))
    progressq.put(len(processed_ids))
    
    small_ds_df["repo_id"] = small_ds_df["id"].str.split("_").str[0]
    small_ds_df.set_index("repo_id", inplace=True)
    repo_ids = sorted(small_ds_df.index.unique(), key=int)
    
    for repo_id in constrained_iterator(sem, repo_ids):
        test_id = small_ds_df.loc[repo_id]["id"]
        
        if not test_id:
            logger.warning(f"No test cases found for repo_id: {repo_id}")
            sem.release()
            continue
        
        if test_id in processed_ids:
            logger.info(f"All test cases for repo_id {repo_id} already processed.")
            sem.release()
            continue
        dataq.put(test_id)

        
    for _ in range(num_proc):
        dataq.put(None)
    for p in process_test_workers:
        p.join()


    writeq.put(None)
    jsonl_writer.join()

    progressq.put(None)
    progress_tracker.join()



def upload_to_hub(output_dir, split, tmp_dir):
    """Upload results to Hugging Face Hub."""

    from datasets import Dataset
    import json
    import pandas as pd
    from datasets import Features, Value, load_dataset


    feat = Features(
        {
        'id': Value(dtype='string', id=None),
        'build_tool': Value(dtype='string', id=None),
        'status': Value(dtype='string', id=None)
        }
    )
    data_path = output_dir / f"buildable_{split}.jsonl"
    runnable_df = pd.read_json(data_path, orient='records', lines=True, dtype=False)
    runnable_df.set_index("id", inplace=True)
    
    dataset_meta = load_dataset("andstor/methods2test_meta", "golden_commit", split=split, cache_dir=tmp_dir / "cache")
    def gen_ds():
        for row in dataset_meta:
            if row["id"] in runnable_df.index:
                runnable_row = runnable_df.loc[row["id"]]
                row["build_tool"] = runnable_row["build_tool"]
                row["status"] = runnable_row["status"]
                yield row
            
            else:
                yield row

    ds = Dataset.from_generator(generator=gen_ds, features=feat, split=split)
    ds.push_to_hub("andstor/methods2test_meta", config_name="test_status", private=False, max_shard_size="250MB")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,  # Or DEBUG for more verbosity
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    args = parse_args()
    
    main(args)
    
    
    if args.push_to_hub:
        upload_to_hub(
            output_dir=SCRIPT_DIR / args.output_dir,
            split=args.split,
            tmp_dir=SCRIPT_DIR / args.tmp_dir
        )