gradient experiments run singlenode `
--name create_classes_test_photos `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py create classes --dataset-name sketchy-test-photos --distance cosine --tsne-dimension 2 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
