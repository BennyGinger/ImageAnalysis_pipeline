FROM nvidia/cuda:11.3.1-base-ubuntu20.04

ARG GIT_USER
ARG GIT_TOKEN

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Install base utilities
RUN apt-get update \
    && apt-get install -y python3-pyqt5 \
    && apt-get install -y build-essential \
    && apt-get install -y git \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone -n https://${GIT_USER}:${GIT_TOKEN}@github.com/BennyGinger/ImageAnalysis_pipeline.git

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Install mamba
RUN conda install -y mamba -c conda-forge

ADD ./environment.yml /root/environment.yml
RUN mamba env update --file /root/environment.yml &&\
    conda clean --all

RUN conda init bash
RUN echo "conda activate cp_dock"  >> ~/.bashrc
ENV PATH /opt/conda/envs/cp_dock/bin:$PATH
ENV CONDA_DEFAULT_ENV $cp_dock
RUN conda run --no-capture-output -n cp_dock python -m pip install cellpose[gui]

# Cellpose gui is running. I'm still getting some errror and warnings, but it seems to be fine.
# Run with: docker run -it --gpus all --name cp_docker -e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix -v "${PWD}:/home" cpdev:v1
