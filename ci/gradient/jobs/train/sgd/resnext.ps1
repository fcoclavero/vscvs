gradient jobs create `
--name train_resnext `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py train --train-validation-split 0.85 --batch-size 50 --epochs 200 --workers 16 --n-gpu 1 ' + `
                         'sgd --learning_rate 0.05 --momentum 0.01 ' + `
                         'resnext --dataset-name sketchy-photos --early-stopping-patience 15 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
