pip install tensorflow==1.12.0 Cython contextlib2 matplotlib pillow lxml

mkdir tod

cd tod

git clone https://github.com/tensorflow/models.git

mv models train_models

cd train_models

git checkout -b 426b2c6e894c22ffb17f32581305ea87c3b8b377

cd ~/tf_ssd/tod/train_models/research

export PYTHONPATH=$PYTHONPATH:/home/opencv-mds/tf_ssd/tod/train_models/research:/home/opencv-mds/tf_ssd/tod/train_models/research/slim

git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI

make

cp -r pycocotools ~/tf_ssd/tod/train_models/research/

-----protocbuf install -------

cd ~/tf_ssd/tod/train_models/research

curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip

unzip protoc-3.2.0-linux-x86_64.zip -d protoc3

sudo mv protoc3/bin/* /usr/local/bin/

sudo mv protoc3/include/* /usr/local/include/

protoc object_detection/protos/*.proto --python_out=.

python object_detection/builders/model_builder_test.py

