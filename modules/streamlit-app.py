from __future__ import annotations

from neuro_sdk import RemoteImage
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from modules.resources import DeployedModelInfo, InferenceServerInfo, ModelStage
from modules.platform_connector import InferenceServerType, InferenceRunner
from modules.mlflow_connector import MLFlowConnector


# general configuration
st.set_page_config(
    page_title="Neuro MLFlow model deployments",
    layout="wide",
    page_icon="",
    menu_items={
        "Get Help": "https://docs.neu.ro/",
        "Report a bug": "https://github.com/neuro-inc/neuro-cli/issues/new/choose",
    }
)
st.header("In-job model deployments")

# Communication
mlflow_conn = MLFlowConnector()
inf_runner = InferenceRunner(
    mlflow_connector=mlflow_conn,
)

# MLFlow registry
models_table = st.container()
models_table.subheader("MLFlow registry")
models = mlflow_conn.get_registered_models()
col1, col2, col3, col4, col5 = models_table.columns([5, 5, 3, 10, 20])
with col1: col1.caption("Model name")
with col2: col2.caption("Stage")
with col3: col3.caption("Version")
with col4: col4.caption("Creation date")
with col5: col5.caption("Deployment")

def deployment_column_entity(model: ModelStage, column: DeltaGenerator) -> None:
    expander: DeltaGenerator = column.expander("Create new deployment")

    deployment_name = expander.text_input(
        "Deployment name",
        value=f"{model.name}-{model.stage}".lower().replace("/", "-").replace("_", "-"),
        max_chars=40,
        key=model,
        help=(
            "Deployment name will be embedded to the resulting model URL. \n"
            "The name can only contain lowercase letters, numbers and hyphens with the following rules:\n"
            "  - the first character must be a letter;\n"
            "  - each hyphen must be surrounded by non-hyphen characters; \n"
            "  - total length must be between 3 and 40 characters long. \n"
        ),
    )
    server_type = InferenceServerType(expander.selectbox("Server type", options=[x.value for x in InferenceServerType], key=model))
    
    if server_type == InferenceServerType.MLFLOW:
        preset_name = expander.selectbox("Preset", options=inf_runner.run_coroutine(inf_runner.list_preset_names()), key=model)
        platform_image = expander.selectbox("Image name", options=inf_runner.run_coroutine(inf_runner.list_images()), key=model)
        image_with_tag = expander.selectbox(
            "Image tag",
            options=inf_runner.run_coroutine(inf_runner.list_image_tags(platform_image)),
            key=model,
            format_func=lambda x: x.tag,
        )
        enable_auth = expander.radio("Force platform Auth", options=[True, False], horizontal=True, key=model)
        expander.button(
            "Deploy",
            key=model,
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
        create_server = expander.radio("Create new server instance", options=[False, True], horizontal=True)
        server_name: str | None = None
        preset_name: str | None = None
        image_with_tag: RemoteImage | None = None
        enable_auth: bool | None = None
        existing_server_info: InferenceServerInfo | None = None
        triton_servers: list[InferenceServerInfo]

        if create_server:
            server_name = expander.text_input("New server name", value="triton", max_chars=40)
            preset_name = expander.selectbox("Preset", options=inf_runner.run_coroutine(inf_runner.list_preset_names()), key=model)
            image_with_tag = expander.selectbox("Image name", options=inf_runner.list_triton_images())
            enable_auth = expander.radio("Force platform Auth", options=[True, False], key=model)
        else:
            triton_servers = [x for x in inf_runner.list_active_inference_servers() if x.type == InferenceServerType.TRITON]
            existing_server_info = expander.selectbox(
                "Select existing server",
                options=triton_servers,
                format_func=lambda x: f"{x.job_name}:{x.job_id}"
            )

        # done with config, validating and allowing to deploy
        input_is_valid = any(
            (
                # deploying to the new triton server
                all(
                    (
                        create_server == True,
                        preset_name,
                        enable_auth is not None,
                        image_with_tag,
                    )
                ),
                # deploying to existing server
                all(
                    (
                        create_server == False,
                        existing_server_info != None,
                    )
                )
            )
        )

        expander.button(
            "Deploy",
            key=model,
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
        col1.write("") # https://github.com/streamlit/streamlit/issues/3052#issuecomment-1083620133
        col1.write(f"[{model.name}]({model.link})")
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
tab_names = [
    "Deployed models",
    "Inference servers"
]
models_tab, servers_tab = deployments_info.tabs(tab_names)

## Models tab
deployed_models = inf_runner.list_all_deployed_models()
model_column_names = DeployedModelInfo.get_md_columns_width().keys()
model_column_widths = DeployedModelInfo.get_md_columns_width().values()
model_columns: list[DeltaGenerator] = models_tab.columns(model_column_widths)

for column_name, column in zip(model_column_names, model_columns):
    column.caption(column_name)
for model in deployed_models:
    model_repr = model.get_md_repr()
    for column_name, column in zip(model_column_names, model_columns):
        column.write(model_repr[column_name])

## Servers tab
active_servers = inf_runner.list_active_inference_servers()
server_column_names = InferenceServerInfo.get_md_columns_width().keys()
server_column_widths = list(InferenceServerInfo.get_md_columns_width().values())
server_column_widths += [3] # kill button
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
        key=server,
        on_click=inf_runner.kill_server,
        args=[server],
    )
