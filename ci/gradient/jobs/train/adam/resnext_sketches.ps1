gradient jobs create `
--name train_resnext_adam_sketches `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py train --train-validation-split 0.9 --batch-size 50 --epochs 100 --workers 16 --n-gpu 1 --tag sketches_pretrained ' + `
                         'adam --learning-rate 0.005 --beta-1 0.9 --beta-2 0.999 --epsilon 0.00000001 --weight-decay 0.0001 --amsgrad True ' + `
                         'resnext --dataset-name sketchy-sketches --pretrained True --early-stopping-patience 5 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
