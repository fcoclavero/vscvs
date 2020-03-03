gradient jobs create `
--name embed_resnext_sketches `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py embed --dataset-name sketchy-sketches --embeddings-name resnext-sketches --batch-size 64 --workers 8 --n-gpu 1 ' + `
                         'resnext --date 20-03-01T16-52 --tag photos --checkpoint net_best_train_6975 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
