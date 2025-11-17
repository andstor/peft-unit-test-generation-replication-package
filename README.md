# PEFT unit test generation replication package

## Repository structure

This repository is organized as follows:
- **/training**: Contains all scripts for training.
- **/generation**: Contains the scripts for generation.
- **/data**: Contains the experiments' generated data.
- **/evaluation**: Contains the scripts for generating coverage data.
- **/analysis**: Contains all scripts used for data analysis.
- **/figures**: Contains all figures created during data analysis.
- **/tables**: Contains all tables created during data analysis.


## Training
To train the models, we use the `run_train.py` CLI script. The script supports various arguments. See the `training/README.md` file for more information, along with the hyperparameters used in the paper.

```
training/
|-- run_train.py                 Python CLI training script.
|-- utils/                       Directory containing utility scripts.
|-- arguments/                   Directory containing the supported arguments for the training script.
|-- deepspeed_configs/           Directory containing various DeepSpeed configurations.
````


## Generation
To generate the unit tests, we use the `run_gen.py` CLI script. The script supports various arguments. See the `generation/README.md` file for more information.
```
generation/
|-- run_gen.py                   Python CLI generation script.
|-- stopping_criterias/          Directory containing utility scripts.
|-- arguments/                   Directory containing the supported arguments for the generation script.
|-- zero_inference_config.json   DeepSpeed-Inference configuration file.
````


## Data

```
data/
|-- <dataset>/
|   |-- generated/                                     The generated unit tests from each experiment.
|   |   |-- <tuning_method>/                           The tuning method used. Full pre-trained, fine-tuning, LoRA, IA^3, and prompt tuning.
|   |       |-- <namespace>/                           The organization that created the base model.
|   |           |-- <model_name>/                      The name of the model.
|   |               |-- 0000[i]-of-0000[n].test.jsonl  JSONL file with generated unit tests.
|   |-- fixed/                                         Same as "generated" but with fixed data.
|   |-- executed/                                      The execution data of the generated unit tests.
|   |   |-- <tuning_method>/                           The tuning method used. Full pre-trained, fine-tuning, LoRA, IA^3, and prompt tuning.
|   |       |-- <namespace>/                           The organization that created the base model.
|   |           |-- <model_name>/                      The name of the model.
|   |               |-- results.jsonl                  JSONL file with execution data.
|   |-- codebleu_scores.csv                            CSV file containing the CodeBLEU scores of the experiments.
|   |-- coverage_branch.csv                            CSV file containing the branch coverage of the generated unit tests.
|   |-- coverage_instruction.csv                       CSV file containing the instruction coverage of the generated unit tests.
|   |-- mutation_score.csv                             CSV file containing the mutation scores of the generated unit tests.
|   |-- passing_rate.csv                               CSV file containing the percentage of the generated unit tests that are runnable.
|   |-- valid_syntax.csv                               CSV file containing the valid syntax fraction generated code.
|-- model_downloads.csv                                CSV file with model download statistics.
|-- params_data.csv                                    CSV file with count of trainable parameters for each model.
```


## Analysis
From the generated data, we fix it using the `fix_data.ipynb` notebook. After fixing the data, we calculate the CodeBLEU scores using the `calc_similarity.ipynb` notebook. After code coverage and mutation scores are calculated (see the [evaluation](#evaluation) section), we calculate the statistics of the passing rate, coverage, and mutation score results using the `calc_execution_metrics.ipynb` notebook. Finally, we analyze the data and generate the plots using the `plots.ipynb` notebook.

```
analysis/
|-- java-universal-parser/          Directory containing the Java parser used to validate generated code.
|-- fix_data.ipynb                  Jupyter Notebook file used to fix the generated data. Also calculates the syntactic validity of the generated code.
|-- calc_execution_metrics.ipynb    Jupyter Notebook file used to calculate the statistics of the passing rate, coverage, and mutation score results.
|-- calc_similarity.ipynb           Jupyter Notebook file used to calculate the CodeBLEU scores of the fixed data.
|-- plots.ipynb                     Jupyter Notebook file containing the Python code used to analyze the extracted data and generate the resulting plots.
|-- tables.ipynb                    Jupyter Notebook file containing the Python code used to analyze the extracted data and generate the resulting tables.
```

## Evaluation
Code coverage is calculated using the `evaluate_humaneval-x.py` script. Due to potential security issues with executing arbitrary generated code, we use Docker. Execute at your own risk. See the `evaluation/README.md` file for container build instructions.

````
evaluation/
|-- humaneval-x/                          Directory containing the scripts for evaluating the Humaneval-X codes.
|   |-- Dockerfile                        Dockerfile for building the evaluation environment.
|   |-- evaluate_tests.py                 Python script for evaluating the generated codes.
|   |-- pom.xml                           Maven project file for building the evaluation environment.
|-- methods2test_runnable/                Directory containing the scripts for evaluating the runnable methods2test codes.
|   |-- Dockerfile                        Dockerfile for building the evaluation environment.
|   |-- evaluate_tests.py                 Python script for evaluating the generated codes.
|   |-- validate_buildable.py             Python script for validating the buildable methods2test codes.
|   |-- validate_runnable.py              Python script for validating the runnable methods2test codes.
|   |-- find_golden_commits.py            Python script for finding the golden commits in the methods2test repository.
|   |-- package_runnable.py               Python script for packaging the runnable methods2test codes.
|   |-- src/                              Directory containing the source code for the runnable methods2test codes.
|   |   |-- jacoco_report.py              Python script for extracting the Jacoco report.
|   |   |-- java_descriptor_converter.py  Python script for converting Java descriptors.
|   |   |-- java_utils.py                 Python script for Java utility functions.
|   |   |-- pitester_report.py            Python script for extracting the Pitest report.
|   |   |-- surefire_report.py            Python script for extracting the Surefire report.
|   |   |-- test_executer.py              Python script for executing the tests.
|   |-- output/                           Directory for storing the intermediate results.
|   |   |-- commits_[split].jsonl         JSONL file with commits for buildable methods2test test repositories.
|   |   |-- buildable_[split].jsonl       JSONL file with the buildable methods2test codes.
|   |   |-- runnable_[split].jsonl        JSONL file with the runnable methods2test codes.
````

## Replication
Follow the setup instructions within each directory. To replicate the experiments, each follow the steps below:

1. Train the models using the `run_train.py` script.
2. Construct the `methods2test_runnable` evaluation dataset by following the instructions in the `evaluation/methods2test_runnable/README.md` file.
3. Generate the unit tests for the `methods2test_runnable` dataset and focal methods for the `humaneval-x` dataset using the `run_gen.py` script.
4. Fix the generated data using the `fix_data.ipynb` notebook.
5. Calculate the CodeBLEU scores using the `calc_similarity.ipynb` notebook.
6. Execute tests and collect quality metrics data by following instructions for running evaluation of the `methods2test_runnable` dataset and the `humaneval-x` dataset. See the respective `README.md` files for details.
7. Calculate the statistics of the passing rate, coverage, and mutation score results using the `calc_execution_metrics.ipynb` notebook.
8. Analyze the data and generate the figures using the `plots.ipynb` notebook.
9. Analyze the data and generate tables by running the `tables.ipynb` notebook.

Due to the variability of deep learning, we provide both the trained models and the generated results. The results are available in the `data/` directory. Metadata and links to the trained models can be found at [here](https://huggingface.co/datasets/andstor/peft-unit-test-generation-experiments). Datasets are available at: [methods2test_small](https://huggingface.co/datasets/andstor/methods2test_small), [methods2test_meta](https://huggingface.co/datasets/andstor/methods2test_meta), [methods2test_runnable](https://huggingface.co/datasets/andstor/methods2test_runnable).
