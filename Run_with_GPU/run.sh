#export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
echo "start training"
unzip imagenet_train_val_csv.zip -d ./
python ./Generate_EBV/Generate_EBV.py
python ./ImageNet_Validation_Experiment/EBV_ImageNet.py
