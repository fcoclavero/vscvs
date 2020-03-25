gradient jobs create `
--name train_siamese_resnext_adam `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'python main.py train --train-validation-split 0.9 --batch-size 16 --epochs 50 --workers 16 --n-gpu 1 ' + `
                         'adam --learning-rate 0.005 --beta-1 0.9 --beta-2 0.999 --epsilon 0.00000001 --weight-decay 0.0001 --amsgrad True ' + `
                         'siamese --dataset-name sketchy --margin 0.6 resnext && ' + `
    'sh ci/gradient/cleanup.sh' `
)
