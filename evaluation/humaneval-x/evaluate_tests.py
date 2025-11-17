
import pandas as pd
import os
from pathlib import Path
import json
import subprocess
from datasets import load_dataset
import argparse
from src.surefire_report import SurefireReportParser

import logging
logger = logging.getLogger(__name__)


SCRIPT_PATH: str = Path(os.path.abspath(__file__))
SCRIPT_DIR: str = SCRIPT_PATH.parent




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate generated tests on methods2test dataset.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--data_dir", type=str, default="../../data/humaneval-x/fixed", help="Temporary directory for caching and repos. Relative to script directory.")
    parser.add_argument("--output_dir", type=str, default="../../data/humaneval-x/executed", help="Directory to store results. Relative to script directory.")
    parser.add_argument("--tmp_dir", type=str, default="tmp", help="Temporary directory for caching and repos. Relative to script directory.")

    return parser.parse_args()


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
    
    dataset = load_dataset("zai-org/humaneval-x", "java", revision="refs/pr/5")
    humanevalx = dataset["test"].to_pandas().set_index("task_id")
    
    file_paths = find_file_paths(data_dir)
    
    for file_path in file_paths:
        res_file_dir = save_dir / Path(*file_path.split(os.sep)[-4:-1]) # Extract method, namespace, and model name from the path
        os.makedirs(res_file_dir, exist_ok=True)
        output_file = res_file_dir / "jacoco.jsonl"
        
        gen_ds_df = pd.read_json(file_path, orient='records', lines=True, dtype=False)  
        if gen_ds_df.empty:
            # Generate an empty output file if not exists
            if not os.path.exists(output_file):
                with open(output_file, "w") as f:
                    pass
            continue
        
        for i, row in gen_ds_df.iterrows():
            test_id = row["id"]
            solution = humanevalx.loc[test_id]["declaration"] + row["prediction"]

            package = "package com.humaneval;\n"

            imports = "import org.junit.*;\nimport static org.junit.Assert.*;"
            imports += "\n" + solution.split("class")[0]

            new_classname = "Solution" + test_id.replace("/", "")
            solution = solution.replace("Solution", new_classname)

            main_path = Path("src") / "main" / "java" / "com" / "humaneval" / (new_classname + ".java")
            os.makedirs(main_path.parent, exist_ok=True)
            with open(main_path, "w") as f:
                f.write(package + "\n" + solution)

            test = humanevalx.loc[test_id]["test"]
            test = test.replace("Solution", new_classname)
            main_preamble = "public class Main {\n    public static void main(String[] args)"
            test_preamble = "public class " + new_classname + "Test {\n	\n	@Test\n	public void test" + test_id.replace("/", "") + "()"
            test = test.replace(main_preamble, test_preamble)
            
            main_class_preamble = "public class Main"
            new_main_class_preamble = "public class " + new_classname + "Test"
            test = test.replace(main_class_preamble, new_main_class_preamble)

            main_method_preamble = "public static void main(String[] args)"
            new_main_method_preamble = "@Test\n	public void test" + test_id.replace("/", "") + "()"
            test = test.replace(main_method_preamble, new_main_method_preamble)

            test_path = Path("src") / "test" / "java" / "com" / "humaneval" / (new_classname + "Test.java")
            os.makedirs(test_path.parent, exist_ok=True)
            with open(test_path, "w") as f:
                f.write(package + "\n" + imports + "\n" + test)
            
            try:
                result = subprocess.run(["mvn", "clean", "test", "-Dmaven.test.failure.ignore=true"], timeout=120)
            except Exception as e:
                data = {}
                data["id"] = test_id
                data["status"] = "exception"
                with open(output_file, "a") as jacoco_file:
                    jacoco_file.write(json.dumps(data) + "\n")
                
            else:
                parser = SurefireReportParser()
                results = parser.get_testsuite_results_by_name("com.humaneval." + new_classname + "Test")
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
                data["id"] = test_id
                data["status"] = status

                print(data)
                with open(output_file, "a") as jacoco_file:
                    jacoco_file.write(json.dumps(data) + "\n")
            finally:
                os.remove(test_path)
                os.remove(main_path)

    
    
    
    
    
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,  # Or DEBUG for more verbosity
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    args = parse_args()
    
    main(args)