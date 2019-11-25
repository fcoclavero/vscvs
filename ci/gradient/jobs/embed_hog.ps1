gradient jobs create `
--name embed_hog `
--optionsFile config.yaml `
--command `
    'sh ci/gradient/setup.sh && `
     python main.py embed --dataset-name sketchy-sketches --embeddings-name hog-sketches --batch-size 64 `
                          --workers 10 --n-gpu 1 hog --in-channels 3 --cell-size 24 --bins 9 `
                          --signed-gradients False && `
     sh ci/gradient/cleanup.sh'