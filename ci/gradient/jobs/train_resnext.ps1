gradient jobs create `
--name train_resnext `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py train --train-validation-split 0.9 --batch-size 64 --epochs 50 --workers 16 --n-gpu 1 ' + `
                         'resnext --dataset-name sketchy-photos --lr 0.00001 --momentum 0.9 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
