echo INFO: cleanup
rsync --recursive --update data /storage/vscvs
ls
echo INFO: cleanup complete
