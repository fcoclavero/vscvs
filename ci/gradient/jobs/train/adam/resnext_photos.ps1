gradient experiments run singlenode `
--name train_resnext_adam_photos `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py train --train-validation-split 0.9 --batch-size 50 --epochs 100 --workers 16 --n-gpu 1 --tag photos ' + `
                         'adam --learning-rate 0.001 --beta-1 0.9 --beta-2 0.999 --epsilon 0.000000001 --weight-decay 0.0005 --amsgrad True ' + `
                         'resnext --dataset-name sketchy-photos --pretrained True --early-stopping-patience 10 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
