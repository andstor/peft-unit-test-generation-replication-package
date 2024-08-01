FROM linuxcontainers/alpine:latest

# Install OpenJDK 11 and necessary tools
RUN apk add --no-cache openjdk11-jdk maven git

# Set environment variables
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk
ENV PATH $JAVA_HOME/bin:$PATH

# Clone your Java project from Git
RUN git clone <your_git_repository_url>

# Set the working directory inside the container
WORKDIR /path/to/your/java/project

# Build your Java project with Maven
RUN mvn clean install

# Run JaCoCo to generate code coverage reports
CMD mvn jacoco:prepare-agent test jacoco:report