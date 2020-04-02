gradient experiments run singlenode `
--name embed_hog_photos `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py embed --dataset-name sketchy-photos --embeddings-name hog-photos --batch-size 64 --workers 8 --n-gpu 1 ' + `
                         'hog --in-channels 3 --cell-size 24 --bins 9 --signed-gradients False && ' + `
    'sh ci/gradient/cleanup.sh' `
)
