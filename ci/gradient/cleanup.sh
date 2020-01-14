echo INFO: cleanup
rsync --recursive --update data /storage/vscvs
find data | while read file; do echo "$file"; target="Workspace/Python/Tesis/paperspace/"$file; /storage/other/dbxcli put "$file" "$target"; done
echo INFO: cleanup complete
