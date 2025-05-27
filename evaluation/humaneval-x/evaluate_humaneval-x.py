
import pandas as pd
import os
from pathlib import Path
import json
import subprocess
from datasets import load_dataset

SCRIPT_PATH: str = Path(os.path.abspath(__file__))
SCRIPT_DIR: str = SCRIPT_PATH.parent.parent
DATA_DIR = SCRIPT_DIR / "data"
SAVE_DIR = SCRIPT_DIR / "coverage"

dataset = load_dataset("THUDM/humaneval-x", "java")
humanevalx = dataset["test"].to_pandas().set_index("task_id")

root, dirs, files = next(os.walk(DATA_DIR))
dataset_path = Path(root)

results = []
for method in dirs:
    root, dirs, files = next(os.walk(dataset_path / method))
    for series in dirs:
        root, dirs, files = next(os.walk(dataset_path / method / series))
        for model in dirs:
            root, dirs, files = next(os.walk(dataset_path / method / series / model))
            
            res_file_path = SAVE_DIR / method / series / model / "jacoco.jsonl"
            os.makedirs(res_file_path.parent, exist_ok=True)
            res_file = open(res_file_path, "w")
            error_file_path = SAVE_DIR / method / series / model / "error.jsonl"
            os.makedirs(error_file_path.parent, exist_ok=True)
            error_file = open(error_file_path, "w")

            for file in files:
                if file.endswith(".jsonl"):
                    file_path = dataset_path / method / series / model / file
                    df = pd.read_json(file_path, orient='records', lines=True, dtype=False)
                    print(dataset_path / method / series / model / file)

                    for i, row in df.iterrows():
                        test_id = row["id"]
                        solution = humanevalx.loc[test_id]["declaration"] + row["fixed_prediction"]
                        
                        imports = "import org.junit.*;\nimport static org.junit.Assert.*;"
                        imports += "\n" + solution.split("class")[0]

                        new_classname = "Solution" + test_id.replace("/", "")
                        solution = solution.replace("Solution", new_classname)
                        
                        main_path = Path("src") / "main" / "java" / (new_classname + ".java")
                        os.makedirs(main_path.parent, exist_ok=True)
                        with open(main_path, "w") as f:
                            f.write(solution)

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

                        test_path = Path("src") / "test" / "java" / (new_classname + "Test.java")
                        os.makedirs(test_path.parent, exist_ok=True)
                        with open(test_path, "w") as f:
                            f.write(imports + "\n" + test)
                        
                        try:
                            result = subprocess.run(["mvn", "clean", "test"], timeout=30)
                        except Exception as e:
                            data = {}
                            data["test_id"] = test_id
                            data["reason"] = "timeout"
                            data["method"] = method
                            data["series"] = series
                            data["model"] = model
                            error_file.write(json.dumps(data) + "\n")
                        else:
                            if result.returncode != 0: # Compilation error
                                data = {}
                                data["test_id"] = test_id
                                data["reason"] = "compilation error"
                                data["method"] = method
                                data["series"] = series
                                data["model"] = model
                                error_file.write(json.dumps(data) + "\n")
                            else:
                                jacoco_df = pd.read_csv("target/site/jacoco/jacoco.csv")
                                for i, row in jacoco_df.iterrows():
                                    data = row.to_dict()
                                    data["test_id"] = test_id
                                    data["method"] = method
                                    data["series"] = series
                                    data["model"] = model
                                    if data["INSTRUCTION_COVERED"] + data["INSTRUCTION_MISSED"] == 0:
                                        instruction_coverage = 0
                                    else:
                                        instruction_coverage = data["INSTRUCTION_COVERED"] / (data["INSTRUCTION_MISSED"] + data["INSTRUCTION_COVERED"])
                                    data["instruction_coverage"] = instruction_coverage
                                    
                                    if data["BRANCH_COVERED"] + data["BRANCH_MISSED"] == 0:
                                        branch_coverage = 0
                                    else:
                                        branch_coverage = data["BRANCH_COVERED"] / (data["BRANCH_MISSED"] + data["BRANCH_COVERED"])
                                    data["branch_coverage"] = branch_coverage
                                    res_file.write(json.dumps(data) + "\n")
                        finally:
                            res_file.flush()
                            error_file.flush()
                            os.remove(test_path)
                            os.remove(main_path)
            res_file.close()
            error_file.close()
