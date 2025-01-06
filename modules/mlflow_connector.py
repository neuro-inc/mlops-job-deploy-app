from __future__ import annotations

import datetime as dt
import os
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator

import yaml
from mlflow.deployments import get_deploy_client
from mlflow.tracking import MlflowClient
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from yarl import URL

from modules.resources import ModelInfo, ModelStage, TritonServerInfo


class MLFlowConnector:
    def __init__(self) -> None:
        self.client = MlflowClient()  # URI and token are passed via env vars
        self._mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]
        self._public_mlflow_uri = os.environ.get("MLFLOW_PUBLIC_URI", self._mlflow_uri)

    def get_registered_models(self) -> list[ModelStage]:
        registered_models = self.client.search_registered_models()
        result = []
        for model in registered_models:
            model_versions = self.client.get_latest_versions(
                model.name, stages=["staging", "production"]
            )
            for model_version in model_versions:
                model_stage_record = ModelStage(
                    name=model.name,
                    version=model_version.version,
                    stage=model_version.current_stage or "",
                    creation_datetime=dt.datetime.fromtimestamp(
                        model_version.creation_timestamp
                        / 1000  # stored in milliseconds
                    ),
                    link=URL(
                        f"{self.client._tracking_client.tracking_uri}"
                        f"/#/models/{model.name}/versions/{model_version.version}"
                    ),
                    public_link=URL(
                        f"{self._public_mlflow_uri}"
                        f"/#/models/{model.name}/versions/{model_version.version}"
                    ),
                    uri=URL(f"models:/{model.name}/{model_version.current_stage}"),
                    source=model_version.source,
                    mlmodel_definition=(self.get_model_config(model_version.source)),
                )
                result.append(model_stage_record)
        return result

    @lru_cache
    def get_model_config(self, model_source: str) -> Dict[str, Any]:
        print("Debug: calling get_model_config")
        mlmodel_path = os.path.join(model_source, "MLmodel")
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config_path = Path(
                _download_artifact_from_uri(mlmodel_path, tmpdirname)
            )
            with open(model_config_path) as f:
                return yaml.safe_load(f)

    @contextmanager
    def set_triton_server_cofig(
        self, triton_server_config: TritonServerInfo
    ) -> Iterator[None]:
        # Tricky one, model_repository_path
        # should be shared between
        # the machine where the deployment is triggered
        # and the Triton server machine
        # This is because to deploy model in Triton, one should
        # (1) copy its files to triton server (no API exposed for that)
        # (2) trigger model load via triton server API
        assert triton_server_config.model_repository_path.exists()

        old_triton_url = os.environ.get("TRITON_URL")
        old_triton_model_repo = os.environ.get("TRITON_MODEL_REPO")

        os.environ["TRITON_URL"] = (
            f"http://{triton_server_config.internal_hostname}:"
            f"{triton_server_config.port}"
        )
        os.environ["TRITON_MODEL_REPO"] = str(
            triton_server_config.model_repository_path
        )
        yield

        if old_triton_url:
            os.environ["TRITON_URL"] = old_triton_url
        else:
            os.environ.pop("TRITON_URL")

        if old_triton_model_repo:
            os.environ["TRITON_MODEL_REPO"] = old_triton_model_repo
        else:
            os.environ.pop("TRITON_MODEL_REPO")

    def deploy_triton(
        self,
        model: ModelStage,
        deployment_name: str,
        flavor: str,
        triton_server_config: TritonServerInfo,
    ) -> None:
        with self.set_triton_server_cofig(triton_server_config):
            deploy_client = get_deploy_client("triton")
            deployment: dict = deploy_client.create_deployment(  # type: ignore
                name=deployment_name,
                model_uri=str(model.uri),
                flavor=flavor,
            )
            if deployment.get("name", "") != deployment_name:
                raise RuntimeError(f"Deployment failed: {deployment_name}")

            # TODO: monitor API is ready

    def list_triton_deployments(
        self,
        triton_server_config: TritonServerInfo,
    ) -> list[ModelInfo]:
        assert triton_server_config.model_repository_path.exists()

        with self.set_triton_server_cofig(triton_server_config):
            deploy_client = get_deploy_client("triton")
            triton_models = list(deploy_client.list_deployments())
            results = []
            for model in triton_models:
                results.append(
                    ModelInfo(
                        name=model["name"],
                        # URI is in form models:/<model-name>/<model-stage>
                        stage=model["mlflow_model_uri"].split("/")[-1],
                        version="unknown",
                    )
                )
            return results
