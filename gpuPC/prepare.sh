# 1. 현재 폴더는 아래와 같아야한다.
# cd /tf_ssd/tod/train_models/research

# 2. 파이썬 환경 세팅
export PYTHONPATH=$PYTHONPATH:/tf_ssd/tod/train_models/research:/tf_ssd/tod/train_models/research/slim

# 3. 슬림 환경 설치

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools /tf_ssd/tod/train_models/research/
cd /tf_ssd/tod/train_models/research
curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip
unzip protoc-3.2.0-linux-x86_64.zip -d protoc3
sudo mv protoc3/bin/* /usr/local/bin/
sudo mv protoc3/include/* /usr/local/include/

#3. 슬림 환경 세팅
cd /tf_ssd/tod/train_models/research
protoc object_detection/protos/*.proto --python_out=.
python object_detection/builders/model_builder_test.py
