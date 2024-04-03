FROM nvidia/cuda:11.3.1-base-ubuntu20.04
# Image with cellpose installed and GUI working. Access to NVIDIA GPU.

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Install base utilities
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt install -y python3-pyqt5 \
    && apt-get install -y libxcb-cursor0 \
    && apt-get install -y build-essential \
    && apt-get install -y sudo \
    && apt-get install -y git \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Install mamba
RUN conda install -y mamba -c conda-forge

# Create conda environment
ADD ./environment.yml /root/environment.yml
RUN mamba env create --file /root/environment.yml &&\
    conda clean --all

RUN conda init bash
RUN echo "conda activate cp_dock"  >> ~/.bashrc
ENV PATH /opt/conda/envs/cp_dock/bin:$PATH
ENV CONDA_DEFAULT_ENV $cp_dock
RUN conda run --no-capture-output -n cp_dock python -m pip install cellpose[gui]

# Set up user (in linux this allow to modify the files/folders created by the container in the host machine)
ARG UID
ARG GID
RUN addgroup --gid $GID cpdev && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" cpdev && \
    echo "cpdev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER cpdev

# Cannot clone git repo because of the different user. Only root user can clone the repo.
# Clone git repository
# ENV GIT_USER=$GIT_USER
# RUN git clone -n https://${GIT_USER}:${GIT_TOKEN}@github.com/BennyGinger/ImageAnalysis_pipeline.git ./ImageAnalysis_pipeline
# WORKDIR /ImageAnalysis_pipeline
# RUN git fetch --all
# RUN git checkout ben

# Build with: docker build . --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t cpdev:ben
# Run with: docker run -it --gpus all --name cp_ben -h imaginganalysis  -e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix -v "${PWD}:/home" cpdev:ben
