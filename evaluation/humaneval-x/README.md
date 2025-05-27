# Evaluation

## Description
This directory provides scripts for building a Docker image for executing and calculating coverage of the generated codes. Jacoco is used to calculate the coverage of the generated codes.

## Requirements

### Dependencies
Please make sure you have Docker installed on your machine. See the [Docker installation guide](https://docs.docker.com/get-docker/) for more information.


## Build
Build the image from evaluation/docker/Dockerfile from this directory.

```bash
docker build -t humaneval-x .
```


After obtaining the image, create a folder for storing the results of the evaluation:
```bash
mkdir ../../data/humaneval-x/coverage
```

Start a container using the following command:

```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/../../data/humaneval-x/coverage/,target=/workspace/data/humaneval-x/coverage \
  --mount type=bind,source="$(pwd)"/../../data/humaneval-x/fixed/,target=/workspace/data/humaneval-x/fixed,readonly \
  humaneval-x python evaluate_tests.py
```

Apptainer:
```bash
apptainer run \
  --compat \
  --cwd "/workspace/evaluation/humaneval-x/"  \
  --mount type=bind,source="$(pwd)"/../../data/humaneval-x/coverage/,target=/workspace/data/humaneval-x/coverage \
  --mount type=bind,source="$(pwd)"/../../data/humaneval-x/fixed/,target=/workspace/data/humaneval-x/fixed,readonly \
  docker://ghcr.io/andstor/peft-unit-test-generation-replication-package/humaneval-x:main bash
```


To run the evaluation, run the following script from the root of the workspace directory (please execute with caution, the generated codes might have unexpected behaviors though with very low possibility. Execute at your own risk:

```bash
python evaluate_humaneval-x.py
```

To run the evaluation, run the following script from the root of the workspace directory (please execute with caution, the generated codes might have unexpected behaviors though with very low possibility). Execute at your own risk:
<!--

```bash
python evaluate_tests.py
```
-->