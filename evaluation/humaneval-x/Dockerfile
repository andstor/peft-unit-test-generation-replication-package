# Use python:3.12 as base image
FROM python:3.12

# Install dependencies
ENV PYTHONUNBUFFERED=1
RUN pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor | tee /etc/apt/trusted.gpg.d/adoptium.gpg > /dev/null
RUN echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list
RUN apt-get update
RUN apt-get install temurin-8-jdk -y
#update-alternatives --config java
RUN java -version


# Install Maven 3.8.1
RUN wget https://archive.apache.org/dist/maven/maven-3/3.8.1/binaries/apache-maven-3.8.1-bin.tar.gz
RUN tar -xvzf apache-maven-3.8.1-bin.tar.gz
RUN mv apache-maven-3.8.1 /opt/maven
RUN ln -s /opt/maven/bin/mvn /usr/bin/mvn
RUN rm apache-maven-3.8.1-bin.tar.gz
RUN mvn --version


RUN apt-get clean

WORKDIR /workspace/evaluation/humaneval-x/

# Copy the local Java project folder into the Docker image

COPY  __init__.py .
COPY  src ./src
COPY  evaluate_tests.py .
COPY  pom.xml .
