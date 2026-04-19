# Defects4J Code-Generation Evaluation

## Description
This directory provides scripts for building a Docker image that evaluates
generated method-body predictions against the corresponding fixed bug in
[Defects4J](https://github.com/rjust/defects4j). For every prediction the
evaluator:

1. Looks up the bug metadata (`project_id`, `bug_id`, `class_file`, `method`,
   `target`) via the
   [`andstor/defects4j_fixed_runnable`](https://huggingface.co/datasets/andstor/defects4j_fixed_runnable)
   dataset using the record `id`.
2. Checks out the **fixed** version of the bug.
3. Replaces the original method body with the model prediction.
4. Runs `defects4j compile` followed by `defects4j test`.
5. Records the execution status in
   `data/defects4j_fixed_runnable/executed/<peft>/<namespace>/<model>/results.jsonl`,
   one JSON object per line: `{"id": <int>, "status": "<status>"}`.

Statuses: `success`, `failed`, `compilation error`, `timeout`, `exception`.

The script is resumable — `id`s already present in `results.jsonl` are skipped.

## Requirements

### Dependencies
Make sure Docker is installed. See the
[Docker installation guide](https://docs.docker.com/get-docker/).

## Build
Build the image (`/evaluation/defects4j_fixed_runnable/Dockerfile`) from the current directory:

```bash
docker build -t defects4j_fixed_runnable .
```

## Usage

Create a folder for storing the results of the evaluation:

```bash
mkdir -p ../../data/defects4j_fixed_runnable/executed
```

Start a container using the following command:

> [!CAUTION]
> Generated code is executed inside the container. Run at your own risk.

```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/../../data/defects4j_fixed_runnable/executed/,target=/workspace/data/defects4j_fixed_runnable/executed \
  --mount type=bind,source="$(pwd)"/../../data/defects4j_fixed_runnable/fixed/,target=/workspace/data/defects4j_fixed_runnable/fixed,readonly \
  defects4j_fixed_runnable python evaluate_tests.py
```

### Apptainer
If you want to use Apptainer instead of Docker, we provide pre-built images on
GitHub Container Registry. The image is available at
`ghcr.io/andstor/peft-unit-test-generation-replication-package/defects4j_fixed_runnable:main`.

Because we are executing untrusted code, we recommend using the `--containall`
and `--no-home` flags to prevent the container from accessing your home
directory and other sensitive files. This will require an overlay file to store
intermediate dependencies, checkouts, and the HuggingFace dataset cache.

Create an overlay file:

```bash
apptainer overlay create --size 10240 overlay.img
```

For more information on how to use Apptainer, please refer to the
[Apptainer documentation](https://apptainer.org/docs/user/latest/).

```bash
apptainer run \
  --containall \
  --no-home \
  --overlay overlay.img \
  --cwd "/workspace/evaluation/defects4j_fixed_runnable/" \
  --mount type=bind,source="$(pwd)"/tmp/,target=/workspace/evaluation/defects4j_fixed_runnable/tmp \
  --mount type=bind,source="$(pwd)"/../../data/defects4j_fixed_runnable/executed/,target=/workspace/data/defects4j_fixed_runnable/executed \
  --mount type=bind,source="$(pwd)"/../../data/defects4j_fixed_runnable/fixed/,target=/workspace/data/defects4j_fixed_runnable/fixed,readonly \
  docker://ghcr.io/andstor/peft-unit-test-generation-replication-package/defects4j_fixed_runnable:main python -u evaluate_tests.py --num_proc 1
```
