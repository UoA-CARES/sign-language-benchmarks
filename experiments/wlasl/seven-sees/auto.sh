python wlasl2000.py
bash setup.sh
cd ../../../data/wlasl
cat train_annotations.txt val_annotations.txt > train_annotations.txt
cd ../../experiments/wlasl/seven-sees
python rgb_only.py
python flow_only.py
