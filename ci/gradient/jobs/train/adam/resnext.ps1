gradient jobs create `
--name train_resnext_adam `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py train --train-validation-split 0.9 --batch-size 50 --epochs 200 --workers 16 --n-gpu 1 --tag photos ' + `
                         'adam --learning-rate 0.005 --beta-1 0.9 --beta-2 0.999 --epsilon 0.00000001 --weight-decay 0.0001 --amsgrad True ' + `
                         'resnext --dataset-name sketchy-photos --pretrained True --early-stopping-patience 3 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
