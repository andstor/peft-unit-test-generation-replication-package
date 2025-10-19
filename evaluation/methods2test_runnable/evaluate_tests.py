import shutil
import os
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager, BoundedSemaphore
from src.test_executer import TestExecutor, TestCandidateDescriptor, FocalMethodDescriptor
import queue

import logging
logger = logging.getLogger(__name__)


SCRIPT_PATH: str = Path(os.path.abspath(__file__))
SCRIPT_DIR: Path = SCRIPT_PATH.parent



os.environ["GIT_TERMINAL_PROMPT"] = "0"  # Disable git terminal prompt

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate generated tests on methods2test dataset.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--data_dir", type=str, default="../../data/methods2test_runnable/fixed", help="Temporary directory for caching and repos. Relative to script directory.")
    parser.add_argument("--output_dir", type=str, default="../../data/methods2test_runnable/coverage", help="Directory to store results. Relative to script directory.")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of parallel threads")
    parser.add_argument("--tmp_dir", type=str, default="tmp", help="Temporary directory for caching and repos. Relative to script directory.")

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
def jsonl_writer_worker(writeq: Queue):
    while True:
        try:
            item = writeq.get()
            if item is None:
                break

            try:
                # Get data and its respective file from the queue
                data, file_path = item
                with open(file_path, "a") as jacoco_file:
                    jacoco_file.write(json.dumps(data) + "\n")
                    
            except Exception as e:
                print(f"Error writing data: {e}")
        except queue.Empty:
            continue  
    
def process_test(dataq: Queue, writeq: Queue, progressq: Queue, sem: BoundedSemaphore, tmp_dir):
    logger.info("Starting worker process")
    meta_ds = load_dataset("andstor/methods2test_meta", "golden_commit", split="test", cache_dir=tmp_dir / "cache")
    logger.info("Loaded meta dataset")
    meta_ds_df = meta_ds.to_pandas().set_index("id")
    logger.info("Converted meta dataset to DataFrame")
    raw_ds = load_dataset("andstor/methods2test_raw", split="test", cache_dir=tmp_dir / "cache")
    logger.info("Loaded raw dataset")
    raw_ds_df = raw_ds.to_pandas().set_index("id")
    logger.info("Converted raw dataset to DataFrame")
    
    logger.info("Loaded meta and raw datasets")
    
    
    while True:
        try:
            item = dataq.get()
            if item is None:
                logger.info("No more items to process, exiting worker")
                break
            test_ids, file_path, output_file = item
        except queue.Empty:
            logger.info("No more items to process, exiting worker")
            continue
        
        
        gen_ds_df = pd.read_json(file_path, orient='records', lines=True, dtype=False)  
        gen_ds_df = gen_ds_df.set_index("id")

        from git import Repo
        first_id = next(iter(test_ids))
        sample_raw = raw_ds_df.loc[first_id]
        sample_commit = meta_ds_df.loc[first_id]
        if sample_commit is None:
            logger.warning("No commit found for id:", first_id)
            sem.release()
            continue
        else:
            sample_commit = sample_commit["commit"]
        
        
        repo_url = sample_raw[ "repository"]["url"]
        repo_name = repo_url.split('/')[-1]

        local_dir = tmp_dir / "repos" / str(os.getpid())
        os.makedirs(local_dir, exist_ok=True)
        repo_path = local_dir / repo_name
            
        try:
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
                
                
                
                
            # Main loop to run tests
            for id in test_ids:
                try:
                    sample_raw = raw_ds_df.loc[id]
                    sample_gen = gen_ds_df.loc[id]

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
                    executor.install_coverage_tool()
                    executor.install_mutation_tool()
                    
                    ### GENERATED CODE ###
                    # Now try to compile the generated code
                    gen_body = trim_end_brac(sample_gen["prediction"]).rstrip()
                    orig_body = trim_end_brac(sample_gen["reference"]).rstrip()
                    
                    #executor.clean()
                    executor.replace_test_case(orig_body, gen_body)
                    out, err, returncode = executor.execute()
                    executor.reset_test_class() # Reset the test class to the original state !IMPORTANT!
                    logger.debug(out)
                    logger.debug(err)
                    logger.debug(returncode)
                    
                    
                    results = executor.get_results()
                    status = None
                    if results is None:
                        logger.info("COMPILATION ERROR")
                        status = "compilation error"
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
                    
                    data = {}
                    data["id"] = id
                    data["status"] = status
                    data["build_tool"] = executor.build_system.get_name()
                    
                    coverage = mutation_report = None
                    if results is not None:
                        
                        # add coverage to data without knowing the keys
                        coverage = executor.get_coverage_report()
                        if coverage is not None:
                            data.update(coverage)
                            logger.info(pd.Series(coverage).to_frame().T)
                        
                        # add mutation report to data without knowing the keys
                        mutation_report = executor.get_mutation_report()
                        if mutation_report is not None:
                            data.update(mutation_report)
                            logger.info(f"Number of killed mutants: {len(mutation_report)}")
                    
                    writeq.put((data, output_file))
            
                except Exception as e:
                    data = {}
                    data["id"] = id
                    data["status"] = "exception"
                    data["build_tool"] = executor.build_system.get_name() if executor.build_system else None
                    with open(output_file, "a") as exception_file:
                        exception_file.write(json.dumps(data) + "\n")
                finally:
                    # Ensure we release the semaphore even if an exception occurs
                    progressq.put(1) 

        except Exception as e:
            for id in test_ids:
                data = {}
                data["id"] = id
                data["status"] = "exception"
                data["build_tool"] = None
                with open(output_file, "a") as exception_file:
                    exception_file.write(json.dumps(data) + "\n")
        finally:
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


def find_file_paths(data_dir):
    file_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths

    
def main(args):
    
    tmp_dir = SCRIPT_DIR / args.tmp_dir
    save_dir = SCRIPT_DIR / args.output_dir
    data_dir = SCRIPT_DIR / args.data_dir
    
    file_paths = find_file_paths(data_dir)
    total_tests = sum([sum(1 for line in open(filename)) for filename in file_paths])
    logger.info(f"Total number of test cases: {total_tests}")
    
    
    progressq = Queue()
    progress_tracker = Process(target=progress_tracker_worker, args=(progressq,total_tests))
    progress_tracker.start()
    
    
    
    # Number of shards (write processes)
    writeq = Queue()
    jsonl_writer = Process(target=jsonl_writer_worker, args=(writeq,))
    jsonl_writer.start()
    
    
    sem = BoundedSemaphore(args.num_proc)
    dataq = Queue()
    process_test_workers = [
        Process(target=process_test, args=(dataq, writeq, progressq, sem, tmp_dir), daemon=True)
        for _ in range(args.num_proc)
    ]
    for p in process_test_workers:
        p.start()
    
    
    
    
    for file_path in file_paths:
        res_file_dir = save_dir / Path(*file_path.split(os.sep)[-4:-1]) # Extract method, namespace, and model name from the path
        os.makedirs(res_file_dir, exist_ok=True)
        output_file = res_file_dir / "jacoco.jsonl"
        
        
        gen_ds_df = pd.read_json(file_path, orient='records', lines=True, dtype=False)  
        # Check if loaded json file is empty
        if gen_ds_df.empty:
            # Generate an empty output file if not exists
            if not os.path.exists(output_file):
                with open(output_file, "w") as f:
                    pass
            continue
        
        # Resume logic: collect already processed ids
        processed_ids = set()
        if os.path.exists(output_file):
            out_df = pd.read_json(output_file, orient='records', lines=True, dtype=False)
            if not out_df.empty:
                out_df = out_df.set_index("id")
                processed_ids.update(set(out_df.index))
        progressq.put(len(processed_ids))
        
        gen_ds_df["repo_id"] = gen_ds_df["id"].str.split("_").str[0]
        gen_ds_df.set_index("repo_id", inplace=True)
        repo_ids = sorted(gen_ds_df.index.unique(), key=int)
        
        for repo_id in constrained_iterator(sem, repo_ids):
            test_ids = set(gen_ds_df.loc[[repo_id]]["id"].tolist())
            
            if not test_ids:
                logger.warning(f"No test cases found for repo_id: {repo_id}")
                sem.release()
                continue
            test_ids = test_ids - processed_ids
            
            if len(test_ids) == 0:
                logger.info(f"All test cases for repo_id {repo_id} already processed.")
                sem.release()
                continue
            
            # Sort test_ids to ensure consistent order
            test_ids = sorted(test_ids, key=lambda s: tuple(map(int, s.split('_'))))
            dataq.put((test_ids, file_path, output_file))

        
    for _ in range(args.num_proc):
        dataq.put(None)
    for p in process_test_workers:
        p.join()

    
    writeq.put(None)
    jsonl_writer.join()

    progressq.put(None)
    progress_tracker.join()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.WARNING,  # Or DEBUG for more verbosity
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    args = parse_args()
    
    main(args)