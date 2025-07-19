# Use python:3.12 as base image
FROM python:3.12


COPY requirements.txt /tmp/requirements.txt

# Install dependencies
ENV PYTHONUNBUFFERED=1
RUN pip install --upgrade pip
RUN pip3 install -r /tmp/requirements.txt

# Install maven
RUN apt-get update
RUN apt-get install -y maven
RUN apt-get clean

WORKDIR /workspace/evaluation/humaneval-x/

# Copy the local Java project folder into the Docker image

COPY  __init__.py .
COPY  src ./src
COPY  evaluate_tests.py .
COPY  pom.xml .
