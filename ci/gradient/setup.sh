echo INFO: setup
cp /storage/vscvs.env .env
pip install --upgrade tensorboard && pip install --upgrade torch
nvcc --version
nvidia-smi
pip freeze
echo INFO: setup complete
