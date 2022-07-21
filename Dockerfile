FROM ghcr.io/neuro-inc/base:v22.5.0-runtime

COPY requirements/apt.txt .
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    cat apt.txt | tr -d "\r" | xargs -I % apt-get -qq install --no-install-recommends % && \
    apt-get -qq clean && \
    apt-get autoremove -y --purge && \
    rm -rf apt.txt /var/lib/apt/lists/* /tmp/* ~/*

COPY requirements/python.txt .
RUN pip install --progress-bar=off -U --no-cache-dir -r python.txt && \
    # Installing MLFlow Triton plugin
    git clone --depth=1 --branch v2.23.0 https://github.com/triton-inference-server/server /tmp/triton_server && \
    cd /tmp/triton_server/deploy/mlflow-triton-plugin && \
    python setup.py install && \
    cd /project && \
    rm -rf /tmp/triton_server python.txt
