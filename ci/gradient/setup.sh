echo INFO: setup
cp /storage/vscvs.env .env
conda install -c anaconda cudatoolkit
pip install --upgrade torch
nvcc --version
nvidia-smi
echo INFO: setup complete
