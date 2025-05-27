# Evaluation

## Description
This directory provides scripts for building a Docker image for executing and calculating coverage of the generated codes. Jacoco is used to calculate the coverage of the generated codes.

## Requirements

### Dependencies
Please make sure you have Docker installed on your machine. See the [Docker installation guide](https://docs.docker.com/get-docker/) for more information.


## Build
Build the image from evaluation/docker/Dockerfile from the current directory:

```bash
docker build -t methods2test_runnable .
```


After obtaining the image, create a folder for storing the results of the evaluation:
```bash
mkdir ./output
```


Start a container using the following command:
```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/output/,target=/workspace/evaluation/methods2test_runnable/output \
  methods2test_runnable python validate_runnable.py
```

Apptainer:
```bash
apptainer run \
  --compat \
  --cwd "/workspace/evaluation/methods2test_runnable/"  \
  --mount type=bind,source="$(pwd)"/output/,target=/workspace/data/methods2test_runnable/output \
  docker://ghcr.io/andstor/peft-unit-test-generation-replication-package/methods2test_runnable:main bash
```


After obtaining the image, create a folder for storing the results of the evaluation:
```bash
mkdir ../../data/methods2test_runnable/coverage
```


Start a container using the following command:

```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/../../data/methods2test_runnable/coverage/,target=/workspace/data/methods2test_runnable/coverage \
  --mount type=bind,source="$(pwd)"/../../data/methods2test_runnable/fixed/,target=/workspace/data/methods2test_runnable/fixed,readonly \
  methods2test_runnable python evaluate_tests.py
```

Apptainer:
```bash
apptainer run \
  --compat \
  --cwd "/workspace/evaluation/methods2test_runnable/"  \
  --mount type=bind,source="$(pwd)"/../../data/methods2test_runnable/coverage/,target=/workspace/data/methods2test_runnable/coverage \
  --mount type=bind,source="$(pwd)"/../../data/methods2test_runnable/fixed/,target=/workspace/data/methods2test_runnable/fixed,readonly \
  docker://ghcr.io/andstor/peft-unit-test-generation-replication-package/methods2test_runnable:main bash
```



### Methods2Test_meta

1. Find golden commits (missing from original dataset) using heuristic by executing find_golden_commits.py.
2. Package golden commits in Methods2Test Meta dataset by running "Golden Commit" section of dataset_meta.ipynb.

3. Find runnable tests in Methods2Test_small by executing validate_runnable.py.
4. Package test statuses in Methods2Test Meta dataset by running "Test Statuses" section of dataset_meta.ipynb.

5. Supersample and Package runnable repos from Methods2Test_small with data from Methods2Test by executing supersample.py.

6. Generate test cases for Methods2Test_runnable. See "Generate" folder.

7. Run test coverage for Methods2Test_runnable by executing evaluate_tests.py