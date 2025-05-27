# Use python:3.12 as base image
FROM python:3.12

# Install maven
RUN apt-get update
RUN apt-get install -y maven
RUN apt-get clean

# Copy the local Java project folder into the Docker image
COPY evaluation/humaneval-x /workspace

# Set the working directory inside the container
WORKDIR /workspace

# Install dependencies
ENV PYTHONUNBUFFERED=1
RUN pip install --upgrade pip
RUN pip install -r requirements.txt 
