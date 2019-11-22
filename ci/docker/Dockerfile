# Run from `vscvs/ci/docker` context. Ej: `docker build -t vscvs ci/docker`
FROM pytorch/pytorch:latest
RUN mkdir requirements
COPY /requirements/vscvs.txt /requirements/vscvs.txt
COPY /requirements/textpreprocess.txt /requirements/textpreprocess.txt
COPY /requirements/wordvectors.txt /requirements/wordvectors.txt
RUN pip install -r /requirements/vscvs.txt
RUN pip install -r /requirements/textpreprocess.txt
RUN pip install -r /requirements/wordvectors.txt
CMD echo hello_docker