echo INFO: setup
cp /storage/vscvs.env .env
conda install -y -c anaconda cudatoolkit==10.1
pip install --upgrade torch
nvcc --version
nvidia-smi
echo INFO: setup complete
