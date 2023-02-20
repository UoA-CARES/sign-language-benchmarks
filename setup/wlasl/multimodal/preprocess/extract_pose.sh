# Extract needs the folders to be here so make shortcuts
ln -s ../../../../data/wlasl/rawframes/val val
ln -s ../../../../data/wlasl/rawframes/train train
ln -s ../../../../data/wlasl/rawframes/test test

# Download the checkpoints
cd ViTPose
kaggle datasets download -d sttaseen/vitpose-checkpoint
unzip vitpose-checkpoint.zip

# Run the extract script
cd ..
python extract_pose.py