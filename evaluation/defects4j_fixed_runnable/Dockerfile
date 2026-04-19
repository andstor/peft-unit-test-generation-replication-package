FROM ghcr.io/andstor/defects4j/defects4j:master

#############################################################################
# Install Python 3
#############################################################################

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Make `python` point to python3 (helps with scripts & entrypoints)
RUN ln -sf /usr/bin/python3 /usr/bin/python

#############################################################################
# Install Python dependencies
#############################################################################

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
#############################################################################
# Copy evaluation script
#############################################################################

WORKDIR /workspace/evaluation/defects4j_fixed_runnable/

COPY evaluate_tests.py .