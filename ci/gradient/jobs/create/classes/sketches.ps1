gradient jobs create `
--name create_classes_sketches `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py create classes --dataset-name sketchy-sketches --distance cosine --tsne-dimension 2 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
