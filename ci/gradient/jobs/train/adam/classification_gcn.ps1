gradient experiments run singlenode `
--name train_classification_gcn_adam `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py train --train-validation-split 0.9 --batch-size 50 --epochs 2 --workers 8 --n-gpu 1 --tag binary ' + `
                         'adam --learning-rate 0.005 --beta-1 0.9 --beta-2 0.999 --epsilon 0.00000001 --weight-decay 0.0001 --amsgrad True ' + `
                         'classification-gcn --dataset-name sketchy-test-sketches --processes 8 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
