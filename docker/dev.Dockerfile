# Build from project root: `docker build -t vscvs .`
# To upload to Dockerhub:
# 1) Tag the latest image: `docker tag f0a5d49e616f fcoclavero/vscvs:latest`
# 2) Push to the container registry: `docker push fcoclavero/vscvs`

# To push to the Google Cloud container registry, we need to use the
# following format: `[HOSTNAME]/[PROJECT-ID]/[IMAGE]:[TAG]`.
# 1) Tag the latest image: `docker tag f0a5d49e616f gcr.io/vscvs-283603/vscvs:dev`
# 2) Push to the container registry: `docker push gcr.io/vscvs-283603/vscvs:dev`

FROM fcoclavero/vscvs:latest

COPY .. /vscvs

ENV DATA_DIR=gs://vscvs/vscvs/data
ENV EMBEDDINGS_DIR=gs://vscvs/embeddings
ENV ROOT_DIR=gs://vscvs/vscvs

WORKDIR /vscvs

ENTRYPOINT ["python", "__main__.py"]
