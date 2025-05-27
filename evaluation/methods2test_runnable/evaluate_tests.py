import shutil
import pathlib
import sys
import os
import json
from src.test_executer import JacocoMavenCoverageTool
from pathlib import Path

SCRIPT_PATH: str = Path(os.path.abspath(__file__))
SCRIPT_DIR: Path = SCRIPT_PATH.parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"
SAVE_DIR = DATA_DIR / "methods2test_runnable"  / "coverage"


os.environ["GIT_TERMINAL_PROMPT"] = "0"  # Disable git terminal prompt

#if not (os.environ.get("CONTAINER", "").lower() in ("yes", "y", "on", "true", "1")):



from src.test_executer import TestExecutor

def load_generated_data():

    import os
    import json

    # Define the base directory
    base_dir = "../../data/methods2test_runnable/fixed"

    # Initialize a dictionary to store data
    data = {}

    # Recursively traverse the directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".jsonl"):
                # Extract method, namespace, and model name from the path
                relative_path = os.path.relpath(root, base_dir)
                method, namespace, model_name = relative_path.split(os.sep)
                
                # Initialize structure if not already present
                if method not in data:
                    data[method] = {}
                if namespace not in data[method]:
                    data[method][namespace] = {}
                if model_name not in data[method][namespace]:
                    data[method][namespace][model_name] = []

                # Load JSONL content
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    for line_number, line in enumerate(f, start=1):
                        line = line.strip()  # Remove leading/trailing whitespace
                        if not line:  # Skip empty lines
                            continue
                        try:
                            json_line = json.loads(line)
                            data[method][namespace][model_name].append(json_line)
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON on line {line_number} in file {file_path}: {e}")

    return data

from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path
import subprocess
from src.test_executer import TestExecutor, TestCandidateDescriptor, FocalMethodDescriptor


def trim_end_brac(input_string):
        # Find the last occurrence of '}'
        brace_index = input_string.rfind('}')
        if brace_index != -1:
            # Return the substring up to and including '}'
            return input_string[:brace_index]
        return input_string  # Return the original string if '}' is not found


def main():
    import os
    from datasets import load_dataset
    meta_ds = load_dataset("andstor/methods2test_meta", "golden_commit", split="test", cache_dir=".tmp/cache")
    meta_ds_df = meta_ds.to_pandas().set_index("id")
    raw_ds = load_dataset("andstor/methods2test_raw", split="test", cache_dir=".tmp/cache")
    raw_ds_df = raw_ds.to_pandas().set_index("id")
    
    import pandas as pd
    data = load_generated_data() 
    for method in data:
        for namespace in data[method]:
            for model_name in data[method][namespace]:
                
                res_file_dir = SAVE_DIR / method / namespace / model_name
                os.makedirs(res_file_dir, exist_ok=True)
            
                gen_ds_df = pd.DataFrame(data[method][namespace][model_name]).set_index("id")                
                for id in gen_ds_df.index:
                    from git import Repo

                    sample_gen = gen_ds_df.loc[id]
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
                        data["generated"] = 0
                        # add coverage to data without knowing the keys
                        if coverage is not None:
                            data.update(coverage)
                        with open(res_file_dir / "jacoco.jsonl", "a") as jacoco_file:
                            jacoco_file.write(json.dumps(data) + "\n")
                        
                        if status == "build error":
                            print("Build error in original code")
                            continue
                        if status == "skipped":
                            print("Test skipped in original code")
                            continue
                        
                        
                        
                        ### GENERATED CODE ###
                        # Now try to compile the generated code
                        gen_body = trim_end_brac(sample_gen["fixed_prediction"]).rstrip()
                        orig_body = trim_end_brac(sample_gen["reference"]).rstrip()
                        
                        executor.clean()
                        executor.replace_test_case(orig_body, gen_body)
                        out, err, returncode = executor.execute()
                        executor.reset_test_class() # Reset the test class to the original state

                        print(out)
                        print(err)
                        print(returncode)
                        
                        
                        results = executor.get_results()
                        status = None
                        if results is None:
                            print("COMPILATION ERROR")
                            status = "compilation error"
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
                        try:
                            if os.path.exists(repo_path):
                                shutil.rmtree(repo_path, ignore_errors=True)
                        except BaseException:
                            pass


if __name__ == "__main__":
    main()