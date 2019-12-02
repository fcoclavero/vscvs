gradient jobs create `
--name train_resnet `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py train --train-validation-split 0.9 --batch-size 32 --epochs 10 --workers 6 --n-gpu 1 ' + `
                         'resnet --dataset-name sketchy-photos --lr 0.00001 --momentum 0.9 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
