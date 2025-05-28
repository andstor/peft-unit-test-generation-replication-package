import shutil
import os
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
from threading import BoundedSemaphore
from src.test_executer import TestExecutor, TestCandidateDescriptor, FocalMethodDescriptor

import logging
logger = logging.getLogger(__name__)


SCRIPT_PATH: str = Path(os.path.abspath(__file__))
SCRIPT_DIR: Path = SCRIPT_PATH.parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data" / "methods2test_runnable" / "fixed"
SAVE_DIR = SCRIPT_DIR.parents[1] / "data" / "methods2test_runnable" / "coverage"


os.environ["GIT_TERMINAL_PROMPT"] = "0"  # Disable git terminal prompt


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
    
    pbar = tqdm(desc="Executing tests", total=total, dynamic_ncols=True)
    while True:
        item = progressq.get()
        if item is None:
            break
        pbar.update(1)

def initializer():
    global meta_ds_df, raw_ds_df, failed
    try:
        failed = False
        meta_ds = load_dataset("andstor/methods2test_meta", "golden_commit", split="test", cache_dir=".tmp/cache")
        meta_ds_df = meta_ds.to_pandas().set_index("id")
        raw_ds = load_dataset("andstor/methods2test_raw", split="test", cache_dir=".tmp/cache")
        raw_ds_df = raw_ds.to_pandas().set_index("id")
    except:
        failed = True
    

def process_test(args):
    global meta_ds_df, raw_ds_df, failed
    file_path, progressq = args
    if failed:
        # skip when initializer failed
        return

    with open(file_path, "r") as f:
        df = pd.read_json(file_path, orient='records', lines=True, dtype=False)
        gen_ds_df = df.set_index("id")
        
        res_file_dir = SAVE_DIR / Path(*file_path.split(os.sep)[-4:-1]) # Extract method, namespace, and model name from the path
        os.makedirs(res_file_dir, exist_ok=True)
        output_file = res_file_dir / "jacoco.jsonl"

        # --- Resume logic: collect already processed ids ---
        processed_ids = set()
        incomplete_ids = set()
        if os.path.exists(output_file):
            out_df = pd.read_json(output_file, orient='records', lines=True, dtype=False).set_index("id")
            # Drop all entries with duplicated ids (keep only unique ids)
            out_df_valid = out_df[~out_df.index.duplicated(keep=False)]
            # Drop entries with status in ["exception", "skipped", "build error"]
            out_df_valid = out_df_valid[~out_df_valid["status"].isin(["exception", "skipped", "build error"])]
            incomplete_ids.update(out_df_valid.index)
            # remove incomplete ids from processed_ids
            processed_ids.update(set(out_df.index) - incomplete_ids)
        # ---------------------------------------------------

        for id in gen_ds_df.index:
            if id in processed_ids and id not in incomplete_ids:
                progressq.put(1)  # Signal progress for already processed ids
                continue
            
            from git import Repo
            sample_gen = gen_ds_df.loc[id]
            sample_raw = raw_ds_df.loc[id]
            try:
                sample_commit = meta_ds_df.loc[id]
                if sample_commit is None:
                    #TODO: log failure
                    logger.warning("No commit found for id:", id)
                    continue
                else:
                    sample_commit = sample_commit["commit"]
            except KeyError:
                logger.warning("No commit found for id:", id)
                continue
            
            repo_url = sample_raw[ "repository"]["url"]
            repo_name = repo_url.split('/')[-1]

            local_dir = Path(".tmp") / "repos" / str(os.getpid())
            os.makedirs(local_dir, exist_ok=True)
            repo_path = local_dir / repo_name
            try:
                #check if the repo is already cloned
                if os.path.exists(repo_path):
                    repo = Repo(repo_path)
                else:
                    repo = Repo.clone_from(repo_url, repo_path)
                
                repo.git.reset("--hard")
                repo.git.checkout(sample_commit)
                
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

                if id not in incomplete_ids:
                    # First try to compile the original code
                    executor.clean()
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

                    data = {}
                    data["id"] = id
                    data["status"] = status
                    data["generated"] = 0
                    # add coverage to data without knowing the keys
                    if coverage is not None:
                        data.update(coverage)
                    with open(res_file_dir / "jacoco.jsonl", "a") as jacoco_file:
                        jacoco_file.write(json.dumps(data) + "\n")
                    
                    if status == "build error":
                        logger.info("Build error in original code")
                        continue
                    if status == "skipped":
                        logger.info("Test skipped in original code")
                        continue
                
                
                
                ### GENERATED CODE ###
                # Now try to compile the generated code
                gen_body = trim_end_brac(sample_gen["fixed_prediction"]).rstrip()
                orig_body = trim_end_brac(sample_gen["reference"]).rstrip()
                
                executor.clean()
                executor.replace_test_case(orig_body, gen_body)
                out, err, returncode = executor.execute()
                executor.reset_test_class() # Reset the test class to the original state

                #print(out)
                #print(err)
                #print(returncode)
                
                
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
                
                coverage = None
                if results is not None:
                    coverage = executor.get_coverage_report()
                    logger.info(pd.Series(coverage).to_frame().T)


                data = {}
                data["id"] = id
                data["status"] = status
                data["generated"] = 1
                # add coverage to data without knowing the keys
                if coverage is not None:
                    data.update(coverage)
                with open(res_file_dir / "jacoco.jsonl", "a") as jacoco_file:
                    jacoco_file.write(json.dumps(data) + "\n")
                
            
            except Exception as e:
                data = {}
                data["id"] = id
                data["status"] = "exception"

                with open(res_file_dir / "jacoco.jsonl", "a") as exception_file:
                    exception_file.write(json.dumps(data) + "\n")
                
            finally:
                progressq.put(1)  # Signal progress
                try:
                    if os.path.exists(repo_path):
                        shutil.rmtree(repo_path, ignore_errors=True)
                    # Also cleanup the temp dir for this process id if empty
                    if os.path.exists(local_dir) and not os.listdir(local_dir):
                        shutil.rmtree(local_dir, ignore_errors=True)
                except BaseException:
                    pass

def find_file_paths():
    file_paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths

    
def main(args):
    num_proc = args.num_proc
    file_paths = find_file_paths()
    total_tests = sum([sum(1 for line in open(filename)) for filename in file_paths])
    logger.info(f"Total number of test cases: {total_tests}")
    
    m = Manager()
    
    progressq = m.Queue()
    progress_tracker = Process(target=progress_tracker_worker, args=(progressq,total_tests))
    progress_tracker.start()
    
    with mp.Pool(processes=num_proc, initializer=initializer) as pool:
        sem = mp.BoundedSemaphore(num_proc)
        data_tuples = ((file_path, progressq) for file_path in file_paths)
        for _ in pool.imap_unordered(process_test, constrained_iterator(sem, data_tuples)):
            sem.release()

    progressq.put(None)
    progress_tracker.join()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.WARNING,  # Or DEBUG for more verbosity
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    parser = argparse.ArgumentParser(description="Evaluate tests with optional multiprocessing.")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of processes to use (default: 1)")
    args = parser.parse_args()
    
    main(args)