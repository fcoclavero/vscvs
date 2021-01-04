# Build from project root: `docker build -f docker/Dockerfile -t vscvs .`
# To upload to Dockerhub:
# 1) Tag the latest image: `docker tag f0a5d49e616f fcoclavero/vscvs:latest`
# 2) Push to the container registry: `docker push fcoclavero/vscvs`

FROM nvidia/cuda:11.0-runtime-ubuntu20.04

# Set locale to C.UTF-8 to avoid problems with pytorch-geometric

USER root
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install rsync to move checkpoints and log files to persitent storage

RUN apt-get update && \
    apt-get install -y build-essential rsync wget

# Install miniconda

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -p /miniconda -b
RUN rm miniconda.sh
ENV PATH=/miniconda/bin:${PATH}

# Install project requirements

RUN conda install -y pytorch torchvision ignite cudatoolkit=11.0 -c pytorch

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

# Module requirements

COPY modules/textpreprocess/requirements.txt /textpreprocess.txt
COPY modules/wordvectors/requirements.txt /wordvectors.txt
RUN pip install -r /textpreprocess.txt && \
    pip install -r /wordvectors.txt
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('punkt')" && \
    python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"

# Project requirements

RUN CUDA="cu110" && \
    TORCH="torch-1.7.0" && \
    pip install torch-scatter -f https://pytorch-geometric.com/whl/${TORCH}+${CUDA}.html && \
    pip install torch-sparse -f https://pytorch-geometric.com/whl/${TORCH}+${CUDA}.html && \
    pip install torch-cluster -f https://pytorch-geometric.com/whl/${TORCH}+${CUDA}.html && \
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/${TORCH}+${CUDA}.html && \
    pip install torch-geometric

COPY requirements.txt /vscvs.txt
RUN pip install -r /vscvs.txt
