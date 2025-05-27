import shutil
import os
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from src.test_executer import TestExecutor
from src.test_executer import TestExecutor, TestCandidateDescriptor, FocalMethodDescriptor

os.environ["GIT_TERMINAL_PROMPT"] = "0"  # Disable git terminal prompt

SCRIPT_PATH: str = Path(os.path.abspath(__file__))
SCRIPT_DIR: Path = SCRIPT_PATH.parent



def main():
    raw_ds = load_dataset("andstor/methods2test_raw", split="test", cache_dir=".tmp/cache")
    raw_ds_df = raw_ds.to_pandas().set_index("id")
    small_ds = load_dataset("andstor/methods2test_small", "fm+fc+c+m+f+t+tc", split="test", cache_dir=".tmp/cache")
    meta_ds = load_dataset("andstor/methods2test_meta", "golden_commit", split="test", cache_dir=".tmp/cache")
    meta_ds_df = meta_ds.to_pandas().set_index("id")
    


    res_file_dir = SCRIPT_DIR / "output"
    os.makedirs(res_file_dir, exist_ok=True)

    for row in small_ds:
        id = row["id"]
        from git import Repo
        sample_raw = raw_ds_df.loc[id]
        try:
            sample_commit = meta_ds_df.loc[id]
            if sample_commit is None:
                #TODO: log failure
                print("No commit found for id:", id)
                continue
            else:
                sample_commit = sample_commit["commit"]
        except KeyError:
            print("No commit found for id:", id)
            continue
        
        repo_url = sample_raw[ "repository"]["url"]
        repo_name = repo_url.split('/')[-1]
        local_dir = ".tmp/repos"
        os.makedirs(local_dir, exist_ok=True)
        repo_path = local_dir + "/" + repo_name
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
            # First try to compile the original code
            executor.clean()
            out, err, returncode = executor.execute()
            print(out)
            print(err)
            print(returncode)
            
            
            results = executor.get_results()
            status = None
            if results is None:
                print("BUILD ERROR")
                status = "build error"
            elif results["skipped"] > 0:
                print("TEST SKIPPED")
                status = "skipped"
            elif results["failures"] > 0:
                print("TEST FAILED")
                status = "failed"
            elif results["errors"] > 0:
                print("TEST ERROR")
                status = "error"
            else:
                status = "success"
                print("TEST PASSED")
            
            
            coverage = None
            if results is not None:
                coverage = executor.get_coverage_report()
                print(pd.Series(coverage).to_frame().T)

            data = {}
            data["id"] = id
            data["status"] = status
            data["build_tool"] = executor.build_system.get_name()
            # add coverage to data without knowing the keys
            if coverage is not None:
                data.update(coverage)
            with open(res_file_dir / "runnable.jsonl", "a") as jacoco_file:
                jacoco_file.write(json.dumps(data) + "\n")
            
            if status == "build error":
                print("Build error in original code")
                continue
            if status == "skipped":
                print("Test skipped in original code")
                continue
            
            
        
        except Exception as e:
            data = {}
            data["id"] = id
            data["status"] = "exception"
            data["build_tool"] = executor.build_system.get_name() if executor.build_system else None

            with open(res_file_dir / "runnable.jsonl", "a") as exception_file:
                exception_file.write(json.dumps(data) + "\n")
                
        finally:
            try:
                if os.path.exists(repo_path):
                    shutil.rmtree(repo_path, ignore_errors=True)
            except BaseException:
                pass


if __name__ == "__main__":
    main()