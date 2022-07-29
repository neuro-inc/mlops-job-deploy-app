1. Build application container image:
    - `neuro-flow build app`
2. Run application:
    - `neuro-flow run app    

TODOs:
- Expose proper Triton model endpoint URIs
- Add ability to remove model deployments
- Monitor model endpoint health
- Support deployments to Seldon with [mlflow2seldon](https://github.com/neuro-inc/mlops-k8s-mlflow2seldon)
- Add documentation
    - pitfalls with deployment to mlflow:
        - container image should contain conda available via Path
        - container image should contain mlflow[extras]>=1.25 installed in it
    - pitfalls with deployment to mlflow:
        - onnx only ML models are supported
        - container image should be built on the compatible CUDA drivers
