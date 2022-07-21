from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
import datetime as dt

from enum import Enum
from pathlib import Path
from typing import NamedTuple
from neuro_sdk import JobDescription, JobStatus
from traitlets import default

from yarl import URL


@dataclass(frozen=True)
class ModelStage:
    name: str
    version: str
    stage: str
    creation_datetime: dt.datetime
    uri: URL    # models:/<model-name>/<model-stage>
    link: URL   # https://<mlflow-endpoint>/#/models/<model-name>/versions/<version>


@dataclass
class ModelInfo:
    # todo: merge with ModelStage
    name: str
    stage: str
    version: str


@dataclass
class DeployedModelInfo:
    model_info: ModelInfo
    inference_server_info: InferenceServerInfo

    @staticmethod
    def get_md_columns_width() -> OrderedDict[str, int]:
        # Here we set number of columns to show and their relative width
        result = OrderedDict()
        result["Model Name:Stage:Version"] = 5
        result["Server Type"] = 2
        result["Server Job ID"] = 5
        result["Creation date"] = 3
        result["Endpoint URL"] = 10
        return result


    def get_md_repr(self) -> dict[str, str]:
        return {
            "Model Name:Stage:Version": f"{self.model_info.name}:{self.model_info.stage}:{self.model_info.version}",
            "Server Type": self.inference_server_info.type.value,
            "Server Job ID": f"[{self.inference_server_info.job_id}](https://app.neu.ro/job-details/{self.inference_server_info.job_id})",
            "Creation date": self.inference_server_info.creation_date.strftime("%D %T"),
            "Endpoint URL": f"[{self.inference_server_info.http_url}]({self.inference_server_info.http_url})",
        }


class InferenceServerType(Enum):
    NONE = "none"
    TRITON = "Triton"
    MLFLOW = "MLFlow"
    # BENTOML = "bentoml"


@dataclass
class InferenceServerInfo:
    job_description: JobDescription
    type: InferenceServerType

    @property
    def job_name(self) -> str:
        return self.job_description.name or "<no-name>"

    @property
    def job_tags(self) -> list[str]:
        return self.job_description.tags

    @property
    def http_url(self) -> str:
        return self.job_description.http_url

    @property
    def job_id(self) -> str:
        return self.job_description.id

    @property
    def creation_date(self) -> dt.datetime:
        return self.job_description.history.created_at

    @staticmethod
    def get_md_columns_width() -> OrderedDict[str, int]:
        # Here we set number of columns to show and their relative width
        result = OrderedDict()
        result["Server name"] = 4
        result["Job ID"] = 6
        result["Server Type"] = 3
        result["Preset"] = 3
        result["Owner"] = 4
        result["Creation date"] = 4
        return result


    def get_md_repr(self) -> dict[str, str]:
        return {
            "Server name": self.job_description.name,
            "Job ID": f"[{self.job_description.id}](https://app.neu.ro/job-details/{self.job_description.id})",
            "Server Type": self.type.value,
            "Preset": self.job_description.preset_name,
            "Owner": self.job_description.owner,
            "Creation date": self.job_description.history.created_at.strftime("%D %T"),
        }


@dataclass
class TritonServerInfo(InferenceServerInfo):
    type: InferenceServerType = InferenceServerType.TRITON

    @property
    def port(self) -> int:
        # port where triton management API is running
        return self.job_description.container.http.port

    @property
    def internal_hostname(self) -> str:
        # we might support external HTTP API later, when auth gets supported
        return self.job_description.internal_hostname_named or \
            self.job_description.internal_hostname

    @property
    def model_repository_path(self) -> Path:
        return Path(self.job_description.container.env["TRITON_MODEL_REPO"])
