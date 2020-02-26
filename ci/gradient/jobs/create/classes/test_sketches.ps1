gradient jobs create `
--name create_classes_test_sketches `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'sh ci/gradient/jobs/create/classes/setup.sh && ' + `
    'python main.py create classes --dataset-name sketchy-test-sketches --distance cosine --tsne-dimension 2 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
