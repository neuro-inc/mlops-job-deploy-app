from __future__ import annotations

from typing import Any

import streamlit as st
from apolo_sdk import RemoteImage
from streamlit.delta_generator import DeltaGenerator

from modules.mlflow_connector import MLFlowConnector
from modules.platform_connector import InferenceRunner, InferenceServerType
from modules.resources import DeployedModelInfo, InferenceServerInfo, ModelStage
from modules.version import __version__ as app_ver


# general configuration
st.set_page_config(
    page_title="Neuro MLFlow model deployments",
    layout="wide",
    page_icon="",
    menu_items={
        "Get Help": "https://docs.neu.ro/",
        "Report a bug": "https://github.com/neuro-inc/neuro-cli/issues/new/choose",
        "About": (
            "[Application](https://github.com/neuro-inc/mlops-job-deploy-app) "
            f"version: {app_ver}."
        ),
    },
)
st.header("In-job model deployments")

# Communication
if "mlflow_connector" not in st.session_state:
    print("Not in session state")
    st.session_state["mlflow_connector"] = MLFlowConnector()
mlflow_conn = st.session_state["mlflow_connector"]  # MLFlowConnector()
inf_runner = InferenceRunner(
    mlflow_connector=mlflow_conn,
)

# MLFlow registry
models_table = st.container()
models_table.subheader("MLFlow registry")
models = mlflow_conn.get_registered_models()
col1, col2, col3, col4, col5 = models_table.columns([5, 5, 3, 10, 20])
with col1:
    col1.caption("Model name")
with col2:
    col2.caption("Stage")
with col3:
    col3.caption("Version")
with col4:
    col4.caption("Creation date")
with col5:
    col5.caption("Deployment")

# preloading options
if "image_list_mlflow" not in st.session_state:
    st.session_state["image_list_mlflow"] = inf_runner.run_coroutine(
        inf_runner.list_images(github=True, platform=True)
    )
if "image_list_triton" not in st.session_state:
    st.session_state["image_list_triton"] = inf_runner.run_coroutine(
        inf_runner.list_images(triton=True, platform=True)
    )
if "image_tags" not in st.session_state:
    st.session_state["image_tags"] = {}
if "presets" not in st.session_state:
    st.session_state["presets"] = inf_runner.run_coroutine(
        inf_runner.list_preset_names()
    )


def deployment_column_entity(model: ModelStage, column: DeltaGenerator) -> None:
    expander: DeltaGenerator = column.expander("Create new deployment")

    deployment_name = expander.text_input(
        "Deployment name",
        value=f"{model.name}-{model.stage}".lower().replace("/", "-").replace("_", "-"),
        max_chars=40,
        key="Deployment name:" + str(model),
        help=(
            "Deployment name will be embedded to the resulting model URL. \n"
            "The name can only contain lowercase letters, numbers"
            " and hyphens with the following rules:\n"
            "  - the first character must be a letter;\n"
            "  - each hyphen must be surrounded by non-hyphen characters; \n"
            "  - total length must be between 3 and 40 characters long. \n"
        ),
    )
    server_type = InferenceServerType(
        expander.selectbox(
            "Server type",
            options=[
                x.value
                for x in InferenceServerType
                if x == InferenceServerType.MLFLOW or model.supports_triton()
            ],
            key="Server type:" + str(model),
        )
    )
    preset_name: str | None = None
    enable_auth: bool | None = None
    image_with_tag: RemoteImage | None = None

    if server_type == InferenceServerType.MLFLOW:
        preset_name = expander.selectbox(
            "Preset",
            options=inf_runner.run_coroutine(inf_runner.list_preset_names()),
            key="Preset:" + str(model),
        )
        image_name: Any = expander.selectbox(
            "Image name",
            options=st.session_state["image_list_mlflow"],
            key="Image name:" + str(model),
            help="""Image should contain mlflow[extras]>=1.27.0 and conda
            accessible on PATH in order for mlflow serve to work properly""",
        )
        if not (image_tag := st.session_state["image_tags"].get(image_name)):
            image_tag = inf_runner.run_coroutine(inf_runner.list_image_tags(image_name))
            st.session_state["image_tags"]["image_name"] = image_tag
        image_with_tag = expander.selectbox(
            "Image tag",
            options=image_tag,
            key="Image tag:" + str(model),
            format_func=lambda x: x.tag,
        )
        enable_auth = expander.radio(
            "Force platform Auth",
            options=[True, False],
            horizontal=True,
            key="Force platform Auth:" + str(model),
        )
        expander.button(
            "Deploy",
            key="Deploy:" + str(model),
            on_click=inf_runner.deploy_mlflow,
            kwargs={
                "model": model,
                "deployment_name": deployment_name,
                "display_container": expander,
                "preset_name": preset_name,
                "image_with_tag": image_with_tag,
                "enable_auth": enable_auth,
            },
        )

    elif server_type == InferenceServerType.TRITON:
        expander.caption(
            "Note: your model should be saved with ONNX inference flawor. "
            "We can not verify this automatically yet."
        )
        create_server = expander.radio(
            "Create new server instance", options=[False, True], horizontal=True
        )
        server_name: str | None = None
        existing_server_info: InferenceServerInfo | None = None
        triton_servers: list[InferenceServerInfo]

        if create_server:
            server_name = expander.text_input(
                "New server name", value="triton", max_chars=40
            )
            preset_name = expander.selectbox(
                "Preset",
                options=st.session_state["presets"],
                key="Preset:" + str(model),
            )
            image_name = expander.selectbox(
                "Image name",
                options=st.session_state["image_list_triton"],
                help="Image with Triton server should contain ONNX inference backend",
            )
            if not (image_tag := st.session_state["image_tags"].get(image_name)):
                image_tag = inf_runner.run_coroutine(
                    inf_runner.list_image_tags(image_name)
                )
                st.session_state["image_tags"]["image_name"] = image_tag
            image_with_tag = expander.selectbox(
                "Image tag",
                options=image_tag,
                key="Image tag:" + str(model),
                format_func=lambda x: x.tag,
            )
            enable_auth = expander.radio(
                "Force platform Auth",
                options=[True, False],  # key=str(model)
            )
        else:
            triton_servers = [
                x
                for x in inf_runner.list_active_inference_servers()
                if x.type == InferenceServerType.TRITON
            ]
            existing_server_info = expander.selectbox(
                "Select existing server",
                options=triton_servers,
                format_func=lambda x: f"{x.job_name}:{x.job_id}",
            )

        # done with config, validating and allowing to deploy
        input_is_valid = any(
            (
                # deploying to the new triton server
                all(
                    (
                        create_server is True,
                        preset_name,
                        enable_auth is not None,
                        image_with_tag,
                    )
                ),
                # deploying to existing server
                all(
                    (
                        create_server is False,
                        existing_server_info is not None,
                    )
                ),
            )
        )

        expander.button(
            "Deploy",
            key="Deploy:" + str(model),
            on_click=inf_runner.deploy_triton,
            kwargs={
                "display_container": expander,
                "model": model,
                "deployment_name": deployment_name,
                "create_server": create_server,
                "existing_server_info": existing_server_info,
                "server_name": server_name,
                "preset_name": preset_name,
                "image_with_tag": image_with_tag,
                "enable_auth": enable_auth,
            },
            disabled=not input_is_valid,
        )


for model in models:
    with col1:
        col1.write(
            ""
        )  # https://github.com/streamlit/streamlit/issues/3052#issuecomment-1083620133
        col1.write(f"[{model.name}]({model.public_link})")
        col1.write("")
    with col2:
        col2.write("")
        col2.write(f"{model.stage}")
        col2.write("")
    with col3:
        col3.write("")
        col3.text(f"{model.version}")
        col3.write("")
    with col4:
        col4.write("")
        col4.text(f'{model.creation_datetime.strftime("%D %T")}')
        col4.write("")
    with col5:
        deployment_column_entity(model, col5)

# Deployments information
deployments_info = st.container()
deployments_info.subheader("Deployments information")
tab_names = ["Deployed models", "Inference servers"]
models_tab, servers_tab = deployments_info.tabs(tab_names)

# Models tab
deployed_models = inf_runner.list_all_deployed_models()
model_column_names = DeployedModelInfo.get_md_columns_width().keys()
model_column_widths = DeployedModelInfo.get_md_columns_width().values()
model_columns: list[DeltaGenerator] = models_tab.columns(list(model_column_widths))

for column_name, column in zip(model_column_names, model_columns):
    column.caption(column_name)
for deployed_model in deployed_models:
    model_repr = deployed_model.get_md_repr()
    for column_name, column in zip(model_column_names, model_columns):
        column.write(model_repr[column_name])

# Servers tab
active_servers = inf_runner.list_active_inference_servers()
server_column_names = InferenceServerInfo.get_md_columns_width().keys()
server_column_widths = list(InferenceServerInfo.get_md_columns_width().values())
server_column_widths += [3]  # kill button
server_columns: list[DeltaGenerator] = servers_tab.columns(server_column_widths)

for column_name, column in zip(server_column_names, server_columns):
    column.caption(column_name)
server_columns[-1].caption("Terminate server")

for server in active_servers:
    server_repr = server.get_md_repr()
    for column_name, column in zip(server_column_names, server_columns):
        column.write(server_repr[column_name])
    server_columns[-1].button(
        "Terminate",
        key=str(server),
        on_click=inf_runner.kill_server,
        args=(server,),
    )
