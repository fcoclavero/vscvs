gradient jobs create `
--name train_resnext `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py train --train-validation-split 0.85 --batch-size 50 --epochs 200 --workers 16 --n-gpu 1 ' + `
                         'adam --learning-rate 0.05 --beta-1 0.9 --beta-2 0.999 --epsilon 0.000000008 --weight-decay 0 --amsgrad false ' + `
                         'resnext --dataset-name sketchy-photos --early-stopping-patience 15 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
