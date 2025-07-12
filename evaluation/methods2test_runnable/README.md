# Evaluation

## Description
This directory provides scripts for building a Docker image for executing and calculating coverage of the generated codes. Jacoco is used to calculate the coverage of the generated codes.

## Requirements

### Dependencies
Please make sure you have Docker installed on your machine. See the [Docker installation guide](https://docs.docker.com/get-docker/) for more information.


## Build

### Docker Image
Build the image from evaluation/docker/Dockerfile from the current directory:

```bash
docker build -t methods2test_runnable .
```


## Usage
### Validate Buildable Repos

Start a container using one of the following:

```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/output/,target=/workspace/evaluation/methods2test_runnable/output \
  methods2test_runnable python -u validate_buildable.py --num_proc 1
```

### Validate Runnable Tests

Start a container using one of the following:

```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/output/,target=/workspace/evaluation/methods2test_runnable/output \
  methods2test_runnable python -u validate_runnable.py --num_proc 1
```


### Evaluate Tests

Evaluate the tests using the following command. This will run the tests and calculate the coverage using Jacoco.

```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/.tmp,target=/workspace/evaluation/methods2test_runnable/.tmp \
  --mount type=bind,source="$(pwd)"/../../data/methods2test_runnable/coverage/,target=/workspace/data/methods2test_runnable/coverage \
  --mount type=bind,source="$(pwd)"/../../data/methods2test_runnable/fixed/,target=/workspace/data/methods2test_runnable/fixed,readonly \
  methods2test_runnable python -u evaluate_tests.py --num_proc 1
```





## Apptainer
If you want to use Apptainer instead of Docker, we provide pre-built images on GitHub Container Registry. The image is available at `ghcr.io/andstor/peft-unit-test-generation-replication-package/methods2test_runnable:main`.

Because we are executing untrusted code, we recommend using the `--containall` and `--no-home` flags to prevent the container from accessing your home directory and other sensitive files. This will require the use of an overlay file to store intermediate dependencies and results.

You can create an overlay file using the following command:

```bash
apptainer overlay create --size 10240 overlay.img
```

For more information on how to use Apptainer, please refer to the [Apptainer documentation](https://apptainer.org/docs/user/latest/).


```bash
apptainer run \
  --containall \
  --no-home \
  --overlay overlay.img \
  --cwd "/workspace/evaluation/methods2test_runnable/" \
  --mount type=bind,source="$(pwd)"/.tmp/,target=/workspace/evaluation/methods2test_runnable/.tmp \
  --mount type=bind,source="$(pwd)"/../../data/methods2test_runnable/coverage/,target=/workspace/data/methods2test_runnable/coverage \
  --mount type=bind,source="$(pwd)"/../../data/methods2test_runnable/fixed/,target=/workspace/data/methods2test_runnable/fixed,readonly \
  docker://ghcr.io/andstor/peft-unit-test-generation-replication-package/methods2test_runnable:main python -u evaluate_tests.py --num_proc 20
```

  --mount type=bind,source="$(pwd)"/evaluate_tests.py,target=/workspace/evaluation/methods2test_runnable/evaluate_tests.py \


### Methods2Test_meta


1. Find golden commits (missing from original dataset) using heuristic by executing find_golden_commits.py. => commits_test.jsonl, 
   commits_validation.jsonl
- Creates methods2test_meta commit_candidates subset.
- Creates methods2test_meta golden_commits subset.


2. Find buildable repos in Methods2Test_small by executing validate_buildable.py. => buildable_test.jsonl
- Creates methods2test_meta test_status subset.


3. Find runnable tests of all buildable repos from Methods2Test_small by executing validate_runnable.py. => runnable_test.jsonl
- Updates methods2test_meta test_status subset.


4. Package runnable tests in new dataset methods2test_runnable by executing package_runnable.py. Uses methods2test_meta test_status subset.
- Creates methods2test_runnable dataset.










5. Supersample buildable repos from Methods2Test_small with data from Methods2Test by executing dataset_runnable.py. Uses methods2test_meta test_status subset.
- Creates methods2test_runnable dataset.


4. Generate test cases for Methods2Test_runnable. See "Generate" folder.



4. Package test statuses in Methods2Test Meta dataset by running "Test Statuses" section of dataset_meta.ipynb.



7. Run test coverage for Methods2Test_runnable by executing evaluate_tests.py