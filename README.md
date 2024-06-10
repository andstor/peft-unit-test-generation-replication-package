# ASE 2024 replication package


## Repository structure

This repository is organized as follows:
- **/training**: Contains all scripts for training.
- **/generation**: Contains the scripts for generation.
- **/data**: Contains the experiments' generated data.
- **/analysis**: Contains all scripts used for data analysis.
- **/figures**: Contains all figures created during data analysis.

## Data
```
data/
|-- generated/
|   |-- <tuning_method>         The tuning method used. Full fine-tuning, LoRA, IA^3, and prompt tuning.
|   |   |-- <namespace>         The organization that created the base model.
|   |       |-- <model_name>    The name of the model
|-- meta_data.csv               CSV file containing the meta information about models.
|-- scores_data.csv             CSV file containing the extracted data from all experiments.
```


## Analysis
```
analysis/
|── scripts
|   |-- extracted_data.csv: CSV file containing the extracted data from all 33 primary studies based on our designed comparison framework.
|   |-- plots.ipynb: Jupyter Notebook file containing the Python code used to analyze the extracted data and generate the resulting plots.
|-- java-universal-parser
    |-- extracted_data.xlsx: Same content as the CSV file above but with XLSX formatting intact to improve readability.
```

# License

This software is licensed under the MIT License.
