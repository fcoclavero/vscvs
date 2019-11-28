echo cleanup
ls storage/vscvs/data/embeddings
rsync --recursive --update data storage/vscvs
ls storage/vscvs/data/embeddings
echo cleanup complete
