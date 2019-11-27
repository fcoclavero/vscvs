echo cleanup
ls storage/vscvs/data/embeddings
apt-get update && \
apt-get install rsync && \
rsync --recursive --update data storage/vscvs
ls storage/vscvs/data/embeddings
echo cleanup complete
