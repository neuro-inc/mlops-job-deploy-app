FROM ghcr.io/neuro-inc/base:v22.12.0-runtime

# Installing conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh && \
    bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b -u && \
    ~/miniconda3/bin/conda init bash && \
    rm Miniconda3-py39_4.12.0-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:$PATH"
# Update MLFlow
RUN pip install -U mlflow[extras]==1.27.0
