echo INFO: setup
cp /storage/vscvs.env .env
conda install -c conda-forge cudatoolkit-dev
pip install --upgrade torch
nvcc --version
nvidia-smi
echo INFO: setup complete
