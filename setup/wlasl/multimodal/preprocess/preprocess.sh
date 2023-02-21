# Run the scripts
bash extract_pose.sh

pip install --upgrade torch torchvision

python extract_depth.py
python extract_flow.py

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

