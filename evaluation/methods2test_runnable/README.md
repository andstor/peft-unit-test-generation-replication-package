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

Start a container using the following command:
```bash
docker run \
  -it \
  -v "$(pwd)"output/:/workspace/evaluation/methods2test_runnable/output:rw \
  methods2test_runnable python validate_runnable.py
```


Start a container using the following command:

```bash
docker run \
  -it \
  -v "$(pwd)"/../../data/methods2test_runnable/coverage/:/workspace/data/methods2test_runnable/coverage:rw \
  -v "$(pwd)"/../../data/methods2test_runnable/fixed/:/workspace/data/methods2test_runnable/fixed:ro \
  methods2test_runnable python evaluate_tests.py
```




### Methods2Test_meta

1. Find golden commits (missing from original dataset) using heuristic by executing find_golden_commits.py.
2. Package golden commits in Methods2Test Meta dataset by running "Golden Commit" section of dataset_meta.ipynb.

3. Find runnable tests in Methods2Test_small by executing validate_runnable.py.
4. Package test statuses in Methods2Test Meta dataset by running "Test Statuses" section of dataset_meta.ipynb.

5. Supersample and Package runnable repos from Methods2Test_small with data from Methods2Test by executing supersample.py.

6. Generate test cases for Methods2Test_runnable. See "Generate" folder.

7. Run test coverage for Methods2Test_runnable by executing evaluate_tests.py