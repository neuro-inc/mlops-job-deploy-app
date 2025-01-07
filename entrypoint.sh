#!/usr/bin/env bash

# Authorise MLFlow
export NEURO_CLI_DISABLE_PYPI_VERSION_CHECK=True
export MLFLOW_TRACKING_TOKEN=$(apolo config show-token)

python -m streamlit run /app/modules/streamlit-app.py --server.runOnSave=True
