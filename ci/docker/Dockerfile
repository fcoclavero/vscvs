# Run from `vscvs/ci/docker` context. Ej: `docker build -t vscvs ci/docker`
# To upload to Dockerhub:
# 1) Tag the latest image: `docker tag f0a5d49e616f fcoclavero/vscvs:latest`
# 2) Push to the container registry: `docker push fcoclavero/vscvs`
FROM liamlecun/pytorch_geometric:latest
USER root
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
COPY /requirements/reduced.txt /requirements.txt
RUN apt-get install -y rsync && \
    pip install -r /requirements.txt
