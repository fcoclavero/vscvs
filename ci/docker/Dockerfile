# Run from `vscvs/ci/docker` context. Ej: `docker build -t vscvs ci/docker`
FROM fcoclavero/pytorch_geometric:latest
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
COPY /requirements/reduced.txt /requirements.txt
RUN apt-get update && \
    apt-get install -y rsync && \
    pip install -r /requirements.txt
