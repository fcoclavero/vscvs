find data | while read -r file; do echo "$file"; target="Workspace/Python/Tesis/paperspace/"$file; dbxcli put "$file" "$target"; done
