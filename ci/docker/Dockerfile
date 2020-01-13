# Run from `vscvs/ci/docker` context. Ej: `docker build -t vscvs ci/docker`
# To upload to Dockerhub:
# 1) Tag the latest image: `docker tag f0a5d49e616f fcoclavero/vscvs:latest`
# 2) Push to the container registry: `docker push fcoclavero/vscvs`

FROM nvidia/cuda:latest

# Set locale to C.UTF-8 to avoid problems with pytorch-geometric

USER root
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install rsync to move checkpoints and log files to persitent storage

RUN apt-get update && \
    apt-get install -y rsync wget

# Install miniconda

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -p /miniconda -b
RUN rm miniconda.sh
ENV PATH=/miniconda/bin:${PATH}

# Install project requirements

RUN conda install -y pytorch torchvision ignite cudatoolkit=10.1 -c pytorch

# In order to install pytorch-geometric, the build environment must have GPU access and thus must
# be run with [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). To be able to access the nvidia
# runtime during the build, the `nvidia-container-runtime` must be installed and the default runtime must
# be set to `nvidia` in the `/etc/docker/daemon.json` configuration file:
# {
#     "runtimes": {
#         "nvidia": {
#             "path": "nvidia-container-runtime",
#             "runtimeArgs": []
#         }
#     },
#     "default-runtime": "nvidia"
# }
# See [this link](https://github.com/NVIDIA/nvidia-docker/issues/595#issuecomment-519714769) for further reference.

COPY /requirements/vscvs.txt /requirements.txt
RUN pip install -r /requirements.txt
