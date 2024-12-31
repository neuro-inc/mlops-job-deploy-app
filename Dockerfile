FROM python:3.11.11-bullseye

LABEL org.opencontainers.image.source = "https://github.com/neuro-inc/mlops-in-job-deployments"

COPY requirements/python.txt /tmp/python.txt
RUN pip install --progress-bar=off -U --no-cache-dir -r /tmp/python.txt && \
    # Installing MLFlow Triton plugin
    git clone --depth=1 --branch v2.53.0 https://github.com/triton-inference-server/server /tmp/triton_server && \
    cd /tmp/triton_server/deploy/mlflow-triton-plugin && \
    python setup.py install && \
    rm -rf /tmp/triton_server /tmp/python.txt

WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
COPY modules /app/modules

ENTRYPOINT [ "/app/entrypoint.sh" ]
