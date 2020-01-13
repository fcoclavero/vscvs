echo INFO: setup
cp /storage/vscvs.env .env
nvcc --version
nvidia-smi
python -c "from PIL import Image; Image.open('/storage/data/sketchy/sketch/tx_000000000000/apple/n07739125_1406-6.png')"
echo INFO: setup complete
