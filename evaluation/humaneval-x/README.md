# Evaluation

## Description
This directory provides scripts for building a Docker image for executing HumanEval-X tests.

## Requirements

### Dependencies
Please make sure you have Docker installed on your machine. See the [Docker installation guide](https://docs.docker.com/get-docker/) for more information.

Install the Python dependencies defined in the `requirements.txt`.
```bash
pip install -r requirements.txt
```

## Build
Build the image (/evaluation/humaneval-x/Dockerfile) from the current directory:

```bash
docker build -t humaneval-x .
```

### Usage

After obtaining the image, create a folder for storing the results of the evaluation:
```bash
mkdir ../../data/humaneval-x/executed
```

Start a container using the following command:

> [!CAUTION]
> Please execute with caution! Generated codes might have unexpected behaviors. Execute at your own risk.


```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/../../data/humaneval-x/executed/,target=/workspace/data/humaneval-x/executed \
  --mount type=bind,source="$(pwd)"/../../data/humaneval-x/fixed/,target=/workspace/data/humaneval-x/fixed,readonly \
  humaneval-x python evaluate_tests.py
```


### Apptainer
If you want to use Apptainer instead of Docker, we provide pre-built images on GitHub Container Registry. The image is available at `ghcr.io/andstor/peft-unit-test-generation-replication-package/humaneval-x:main`.

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
  --cwd "/workspace/evaluation/humaneval-x/" \
  --mount type=bind,source="$(pwd)"/tmp/,target=/workspace/evaluation/humaneval-x/tmp \
  --mount type=bind,source="$(pwd)"/../../data/humaneval-x/executed/,target=/workspace/data/humaneval-x/executed \
  --mount type=bind,source="$(pwd)"/../../data/humaneval-x/fixed/,target=/workspace/data/humaneval-x/fixed,readonly \
  docker://ghcr.io/andstor/peft-unit-test-generation-replication-package/humaneval-x:main python -u evaluate_tests.py --num_proc 1
```