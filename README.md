# ICSE 2025 replication package


## Repository structure

This repository is organized as follows:
- **/training**: Contains all scripts for training.
- **/generation**: Contains the scripts for generation.
- **/data**: Contains the experiments' generated data.
- **/evaluation**: Contains the scripts for generating coverage data.
- **/analysis**: Contains all scripts used for data analysis.
- **/figures**: Contains all figures created during data analysis.


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
|   |   |   |-- <namespace>/                           The organization that created the base model.
|   |   |       |-- <model_name>/                      The name of the model.
|   |   |           |-- 0000[i]-of-0000[n].test.jsonl  JSONL file with generated unit tests.
|   |   |-- scores.csv                                 CSV file containing the CodeBLEU scores of the experiments.
|   |   |-- valid_syntax.csv                           CSV file containing the valid syntax fraction generated code.
|   |-- fixed/                                         Same as "generated" but with fixed data.
|   |-- coverage/                                      The coverage data of the generated unit tests.
|-- coverage_branch.csv                                CSV file containing the branch coverage of the generated unit tests.
|-- coverage_instruction.csv                           CSV file containing the instruction coverage of the generated unit tests.
|-- coverage_runnable.csv                              CSV file containing the percentage of the generated unit tests that are runnable.
|-- params_data.csv                                    CSV file with count of trainable parameters for each model.
```


## Analysis
From the generated data, we fix it using the `fix_data.ipynb` notebook. After fixing the data, we calculate the CodeBLEU scores using the `calc_score.ipynb` notebook. After code coverage is calculated (see the [evaluation](#evaluation) section), we calculate the statistics of the coverage results using the `calc_coverage.ipynb` notebook. Finally, we analyze the data and generate the plots using the `plots.ipynb` notebook.
```
analysis/
|-- java-universal-parser/       Directory containing the Java parser used to validate generated code.
|-- fix_data.ipynb               Jupyter Notebook file used to fix the generated data.
|-- calc_coverage.ipynb          Jupyter Notebook file used to calculate the statistics of the coverage results.
|-- calc_score.ipynb             Jupyter Notebook file used to calculate the CodeBLEU scores of the fixed data.
|-- plots.ipynb                  Jupyter Notebook file containing the Python code used to analyze the extracted data and generate the resulting plots.
```

## Evaluation
Code coverage is calculated using the `evaluate_humaneval-x.py` script. Due to potential security issues with executing arbitrary generated code, we use Docker. Execute at your own risk. See the `evaluation/README.md` file for container build instructions.
````
evaluation/
|-- Dockerfile                   Dockerfile for building the evaluation environment.
|-- evaluate_humaneval-x.py      Python script for evaluating the generated codes.
|-- pom.xml                      Maven project file for building the evaluation environment.
````

## Replication
Follow the setup instructions within each directory. To replicate the experiments, each follow the steps below:

1. Train the models using the `run_train.py` script.
2. Generate the unit tests using the `run_gen.py` script.
3. Fix the generated data using the `fix_data.ipynb` notebook.
4. Calculate the CodeBLEU scores using the `calc_score.ipynb` notebook.
5. Generate coverage data by executing the `evaluate_humaneval-x.py` script within the provided Docker container.
6. Calculate the statistics of the coverage results using the `calc_coverage.ipynb` notebook.
5. Analyze the data and generate the plots using the `plots.ipynb` notebook.

Due to the variability of deep learning, we provide both the trained models and the generated results. The results are available in the `data/` directory. The trained models are available [here](https://huggingface.co/fals3/peft-unit-test-generation-experiments). 
