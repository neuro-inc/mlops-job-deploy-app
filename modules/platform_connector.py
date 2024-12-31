from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Coroutine

from apolo_sdk import (
    Client,
    HTTPPort,
    IllegalArgumentError,
    JobDescription,
    JobStatus,
    RemoteImage,
    Volume,
    get,
)
from streamlit.delta_generator import DeltaGenerator
from yarl import URL

from modules.mlflow_connector import MLFlowConnector
from modules.registry_client import RegistryV2Client
from modules.resources import (
    DeployedModelInfo,
    InferenceServerInfo,
    InferenceServerType,
    ModelInfo,
    ModelStage,
    TritonServerInfo,
)


logger = logging.getLogger(__name__)


GH_SUPPORTED_IMAGES = [
    RemoteImage(name="ghcr.io/neuro-inc/mlflow"),
    RemoteImage(name="ghcr.io/neuro-inc/base"),
]


NVCR_SUPPORTED_IMAGES = [
    RemoteImage(name="nvcr.io/nvidia/tritonserver"),
]


class InferenceRunner:
    def __init__(self, mlflow_connector: MLFlowConnector) -> None:
        self._client: Client | None = None
        self._mlflow_connector = mlflow_connector

    def get_server_tags(
        self,
        inference_type: InferenceServerType | None = None,
        model: ModelStage | None = None,
    ) -> list[str]:
        result = [f"in-job-deployments::inference-server"]
        if inference_type:
            result.append(f"server-type::{inference_type.value}")
        if model:
            result.append(f"model-info::{model.name}:{model.stage}:{model.version}")
        return result

    def get_inference_server_type(
        self,
        job_description: JobDescription,
    ) -> InferenceServerType:
        tags = job_description.tags
        for tag in tags:
            if "server-type::" in tag:
                _, server_type = tag.split("::")
                return InferenceServerType(server_type)
        raise ValueError(f"Server info not found in job {job_description.id}")

    def get_mlflow_model_info(
        self,
        server_info: InferenceServerInfo,
    ) -> DeployedModelInfo:
        assert server_info.type == InferenceServerType.MLFLOW

        for tag in server_info.job_tags:
            if "model-info::" in tag:
                _, model_info_tag = tag.split("::")
                name, stage, version = model_info_tag.split(":")
                model_info = ModelInfo(
                    name=name,
                    stage=stage,
                    version=version,
                )
                return DeployedModelInfo(
                    model_info=model_info,
                    inference_server_info=InferenceServerInfo(
                        job_description=server_info.job_description,
                        type=InferenceServerType.MLFLOW,
                    ),
                )
        raise ValueError(f"Model info not found in job {server_info.job_id}")

    def get_triton_model_infos(
        self,
        server_info: InferenceServerInfo,
    ) -> list[DeployedModelInfo]:
        assert server_info.type == InferenceServerType.TRITON
        server_config = TritonServerInfo(server_info.job_description)

        model_infos = self._mlflow_connector.list_triton_deployments(server_config)
        result = [
            DeployedModelInfo(model_info, server_info) for model_info in model_infos
        ]

        return result

    def list_active_inference_servers(self) -> list[InferenceServerInfo]:
        return self.run_coroutine(self._list_active_inference_servers())

    async def _list_active_inference_servers(
        self,
        server_type: InferenceServerType | None = None,
    ) -> list[InferenceServerInfo]:
        target_tags = self.get_server_tags(server_type)
        async with get() as n_client:
            result = []
            async for job_descr in n_client.jobs.list(
                statuses=JobStatus.active_items(),
                tags=target_tags,
            ):
                try:
                    server_type = self.get_inference_server_type(job_descr)
                except ValueError as e:
                    logger.warning(f"Unknown inference server type: {e}")
                else:
                    server_info = InferenceServerInfo(
                        job_description=job_descr,
                        type=server_type,
                    )
                    result.append(server_info)
            return result

    async def list_preset_names(self) -> list[str]:
        async with get() as n_client:
            return list(n_client.config.presets.keys())

    async def list_images(
        self,
        triton: bool = False,
        github: bool = False,
        platform: bool = False,
    ) -> list[RemoteImage]:
        tasks = []
        if triton:
            tasks.append(asyncio.create_task(self._list_triton_images()))
        if github:
            tasks.append(asyncio.create_task(self._list_gh_images()))
        if platform:
            tasks.append(asyncio.create_task(self._list_platform_images()))
        results = []
        for res in await asyncio.gather(*tasks):
            results.extend(res)
        return results

    async def _list_gh_images(self) -> list[RemoteImage]:
        return GH_SUPPORTED_IMAGES

    async def _list_triton_images(self) -> list[RemoteImage]:
        return NVCR_SUPPORTED_IMAGES

    async def _list_platform_images(self) -> list[RemoteImage]:
        async with get() as n_client:
            platform_images = await n_client.images.list()
            return platform_images

    async def list_image_tags(self, image: RemoteImage) -> list[RemoteImage]:
        if image._is_in_apolo_registry:
            async with get() as n_client:
                return list(await n_client.images.tags(image))
        else:
            reg, own, *repo = image.name.split("/")
            reg_cl = RegistryV2Client(base_url=URL("https://" + reg))
            try:
                imgs_with_tags = await reg_cl.list_repo_tags(own, "/".join(repo))
                imgs_with_tags = imgs_with_tags[::-1]  # From older to newer
                if image in NVCR_SUPPORTED_IMAGES:
                    reg = r"\d{1,2}\.\d{1,2}-py3"
                    # Leave only those with ONNX backend
                    imgs_with_tags = list(
                        filter(lambda x: re.fullmatch(reg, x.tag or ""), imgs_with_tags)
                    )
                return imgs_with_tags
            except Exception as e:
                logger.error(f"Unable to fetch image tags for {image}: {e}")
                return []

    def deploy_mlflow(self, *args, **kwargs) -> None:  # type: ignore
        return self.run_coroutine(self._deploy_mlflow(*args, **kwargs))

    def deploy_triton(
        self,
        display_container: DeltaGenerator,
        model: ModelStage,
        deployment_name: str,
        create_server: bool,
        existing_server_info: InferenceServerInfo | None = None,
        server_name: str | None = None,
        preset_name: str | None = None,
        image_with_tag: RemoteImage | None = None,
        enable_auth: bool | None = None,
    ) -> None:
        if create_server:
            assert server_name
            assert preset_name
            assert image_with_tag
            assert isinstance(enable_auth, bool)
            display_container.info("Starting Triton server")
            server_config = self.run_coroutine(
                self._deploy_triton_server(
                    server_name=server_name,
                    preset_name=preset_name,
                    image_with_tag=image_with_tag,
                    enable_auth=enable_auth,
                    display_container=display_container,
                )
            )
        else:
            assert existing_server_info
            server_config = TritonServerInfo(existing_server_info.job_description)

        if not server_config:
            display_container.error("Unable to fetch server config")
        else:
            try:
                display_container.info(f"Deploying {model.uri}")
                self._mlflow_connector.deploy_triton(
                    model=model,
                    deployment_name=deployment_name,
                    flavor="onnx",
                    triton_server_config=server_config,
                )
                display_container.success(f"Model deployed!")
            except TypeError as e:
                if "join() argument must be str, bytes," in e.args[0]:
                    display_container.error(
                        f"Could only deploy Triton or ONNX model "
                        "to Triton inference server. Verify model flawors."
                    )
            except Exception as e:
                display_container.error(f"Unable to deploy model: {e}")

    async def _deploy_mlflow(
        self,
        model: ModelStage,
        deployment_name: str,
        preset_name: str,
        image_with_tag: RemoteImage,
        enable_auth: bool,
        display_container: DeltaGenerator,
    ) -> None:
        display_container.info(
            f"Deploying {model.name}:{model.stage} using MLFlow inference."
        )
        logger.debug(f"Running job with image {image_with_tag}")
        async with get() as n_client:
            try:
                job_descr = await n_client.jobs.start(
                    image=image_with_tag,
                    preset_name=preset_name,
                    shm=True,
                    name=deployment_name,
                    env={
                        "MLFLOW_TRACKING_URI": str(model.link.with_path("")),
                    },
                    entrypoint="/bin/bash",
                    command=(
                        "-c "
                        '"source /root/.bashrc && '
                        f"mlflow models serve -m models:/{model.name}/{(model.stage).lower()} "
                        '--host=0.0.0.0 --port=5000 --env-manager conda"'
                    ),
                    # restart_policy=JobRestartPolicy.ON_FAILURE,
                    http=HTTPPort(5000, requires_auth=enable_auth),
                    tags=self.get_server_tags(InferenceServerType.MLFLOW, model),
                )
            except IllegalArgumentError as e:
                display_container.error(
                    f"Deployment with name {deployment_name} already exists: {e}"
                )
            else:
                display_container.info(f"Created a job {job_descr.id}")
                while job_descr.status.is_pending:
                    job_descr = await n_client.jobs.status(job_descr.id)
                    await asyncio.sleep(0.1)
                display_container.success(f"Started a job {job_descr.id}")
                # TODO: monitor API is ready

    async def _deploy_triton_server(
        self,
        server_name: str,
        preset_name: str,
        image_with_tag: RemoteImage,
        enable_auth: bool,
        display_container: DeltaGenerator,
        port: int = 8000,
    ) -> TritonServerInfo | None:
        async with get() as n_client:
            model_repo_storage = URL(os.environ["TRITON_MODEL_REPO_STORAGE"])
            model_repo_job = f"{os.environ['TRITON_MODEL_REPO']}/"
            Path(model_repo_job).mkdir(parents=True, exist_ok=True)
            try:
                job_descr = await n_client.jobs.start(
                    image=image_with_tag,
                    preset_name=preset_name,
                    shm=True,
                    name=server_name,
                    env={
                        "TRITON_MODEL_REPO": model_repo_job,
                    },
                    volumes=[Volume(model_repo_storage, model_repo_job)],
                    command=(
                        f"/bin/bash -c "
                        '"tritonserver --model-control-mode=explicit '
                        "--strict-model-config=false "
                        f'--model-repository=$TRITON_MODEL_REPO --http-port={port}"'
                    ),
                    # restart_policy=JobRestartPolicy.ON_FAILURE,
                    # HTTP API Triton exposes. Together with HTTP API it exposes
                    #  gRPC (8001) and metrics ports (8002)
                    http=HTTPPort(port, requires_auth=enable_auth),
                    tags=self.get_server_tags(InferenceServerType.TRITON),
                )
            except IllegalArgumentError as e:
                display_container.error(
                    f"Server with name {server_name} already exists: {e}"
                )
                return None
            else:
                display_container.info(f"Created a job {job_descr.id}")
                while job_descr.status.is_pending:
                    job_descr = await n_client.jobs.status(job_descr.id)
                    await asyncio.sleep(0.1)
                display_container.success(f"Started a job {job_descr.id}")

                server_config = TritonServerInfo(job_descr)
                return server_config

    def run_coroutine(self, coro: Coroutine[Any, Any, Any]) -> Any:
        return asyncio.run(coro)

    def list_all_deployed_models(self) -> list[DeployedModelInfo]:
        return self.run_coroutine(self.list_deployed_models())

    async def list_deployed_models(
        self, server_types: list[InferenceServerType] | None = None
    ) -> list[DeployedModelInfo]:
        result = []
        if not server_types:
            server_types = [
                x for x in InferenceServerType if x != InferenceServerType.NONE
            ]
        for server_type in server_types:
            server_infos = await self._list_active_inference_servers(server_type)
            for server_info in server_infos:
                if server_info.type == InferenceServerType.MLFLOW:
                    result.append(self.get_mlflow_model_info(server_info))
                elif server_info.type == InferenceServerType.TRITON:
                    result.extend(self.get_triton_model_infos(server_info))
        return result

    def kill_server(self, server: InferenceServerInfo) -> None:
        return self.run_coroutine(self._kill_server(server))

    async def _kill_server(self, server: InferenceServerInfo) -> None:
        async with get() as n_client:
            await n_client.jobs.kill(server.job_id)
