#****vBox 에서 다시 트레이닝 시작할때***

cd ~/tf_ssd/

# 1. start venv
source ./venvssd/bin/activate

# 2. 파이썬 환경 세팅
export PYTHONPATH=$PYTHONPATH:/home/opencv-mds/tf_ssd/tod/train_models/research:/home/opencv-mds/tf_ssd/tod/train_models/research/slim

#3. 슬림 환경 세팅
cd /home/opencv-mds/tf_ssd/tod/train_models/research
protoc object_detection/protos/*.proto --python_out=.
python object_detection/builders/model_builder_test.py

