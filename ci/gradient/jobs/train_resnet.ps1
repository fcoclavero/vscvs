gradient jobs create `
--name train_resnet `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py train --train-validation-split 0.8 --batch-size 64 --epochs 100 --workers 16 --n-gpu 1 ' + `
        'resnet --dataset-name sketchy-photos --lr 0.00001 --momentum 0.9 && ' + `
    'sh ci/gradient/cleanup.sh' `
)
