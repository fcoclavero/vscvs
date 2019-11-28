# Run from `vscvs/ci/docker` context. Ej: `docker build -t vscvs ci/docker`
FROM pytorch/pytorch:latest
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
COPY /requirements/vscvs.txt /requirements/vscvs.txt
COPY /requirements/textpreprocess.txt /requirements/textpreprocess.txt
COPY /requirements/wordvectors.txt /requirements/wordvectors.txt
RUN apt-get update && \
    apt-get install rsync && \
    mkdir requirements && \
    pip install -r /requirements/vscvs.txt && \
    pip install -r /requirements/textpreprocess.txt && \
    pip install -r /requirements/wordvectors.txt
