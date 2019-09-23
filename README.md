# ssd_traing_export
#######################################################

# This is guide for 10 minutes training on Virtual Box.
So it can NEVER be perfect training !!

#######################################################

# venv setting (you can skip !!)
sudo apt install protobuf-compiler python3-tk -y

pip3 install virtualenv

mkdir tf_ssd

cd tf_ssd

virtualenv venvssd

source venvssd/bin/activate

# 2-0 install package

pip install tensorflow==1.12.0 Cython contextlib2 matplotlib pillow lxml 


$ sudo apt-get update

$ sudo apt-get install python3-tk


# 2-1 models for Train

mkdir tod

cd tod

$ git clone https://github.com/tensorflow/models.git

$ mv models train_models

# 2-2 models for Export

$ git clone https://github.com/tensorflow/models.git

$ mv models export_models

$ cd export_models

$ git checkout ae0a9409212d0072938fa60c9f85740bb89ced7e

* Don't be afraid to see Error !

* please check different branch*
$ cd research

$ ls

$ ls ../../train_models/research/


# 2-3 pycocotools, protocbuf install

-----pycocotools install -----

$ cd ~/tf_ssd/tod/train_models/research

$ export PYTHONPATH=$PYTHONPATH:/home/opencv-mds/tf_ssd/tod/train_models/research:/home/opencv-mds/tf_ssd/tod/train_models/research/slim

$ git clone https://github.com/cocodataset/cocoapi.git

$ cd cocoapi/PythonAPI

$ make

$ cp -r pycocotools ~/tf_ssd/tod/train_models/research/

-------------------------
-----protocbuf install -------

$ cd ~/tf_ssd/tod/train_models/research

$ curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip

$ unzip protoc-3.2.0-linux-x86_64.zip -d protoc3

$ sudo mv protoc3/bin/* /usr/local/bin/

$ sudo mv protoc3/include/* /usr/local/include/

$ protoc object_detection/protos/*.proto --python_out=.

$ python object_detection/builders/model_builder_test.py

------------Just you can see "OK" -> it is ok !!
------------------------------------------------------------------------

# 3-2 tfrecord 준비

sudo cp /media/[공유폴더이름]/tfrecord.zip ./

sudo chmod 777 ./tfrecord.zip

unzip tfrecord.zip

===========================================
# 3-3 train model modify

$ vim ~/tf_ssd/tod/train_models/research/object_detection/samples/configs/ssd_inception_v2_coco.config
line: 151, 152 -> 주석(#) 처리

해당 라인은 transfer learning을 하거나 fine_tuning할 때 사용하므로 현재는 사용하지 않는다.

line: 170,184 -> path설정

해당 라인에 적혀있는 path의 tfrecord를 train하므로 우리데이터셋 경로로 바꿔주자.

170, 184: /home/opencv-mds/tf_ssd/tfrecord/ 여기에 ms만 지우면됨

line: 172,186 -> mscoco_label_map.pbtxt 경로를 설정해줘야 한다. 

172, 186: /home/opencv-mds/tf_ssd/tod/train_models/research/object_detection/data/mscoco_label_map.pbtxt

-------------------------
* 수정된 config 예 : vBox에 coco_val.record 가 없어서 동일하게 coco_train.record 을 사용함.

  169 train_input_reader: {                                           
  170   tf_record_input_reader {                                     
  171     input_path: "/home/opencv-mds/tf_ssd/tfrecord/coco_train.record-?????-of-00100"                                             
  172   }                                                            
  173   label_map_path: "/home/opencv-mds/tf_ssd/tod/train_models/research/object_detection/data/mscoco_label_map.pbtxt"              
  174 } 
--------------------------
  183 eval_input_reader: {                                                                                                               
  184   tf_record_input_reader {                                                                                                       
  185     input_path: "/home/opencv-mds/tf_ssd/tfrecord/coco_train.record-?????-of-00100"                                               
  186   }                                                                                                                               
  187   label_map_path: "/home/opencv-mds/tf_ssd/tod/train_models/research/object_detection/data/mscoco_label_map.pbtxt"                 
  188   shuffle: false                                                                                                                   
  189   num_readers: 1                                                                                                                   
  190 } 
--------------------------

*vBox 의 경우 메모리 여유가 없으므로 136line 의 배치 사이즈를 4로 수정 필요 !!


# 3-4 train

이제 학습에 필요한 파라미터들을 설정해주고 실행하면 된다.

mkdir ~/tf_ssd/tod/save_models/ 

mkdir ~/tf_ssd/tod/save_models/coco_test

$ PIPELINE_CONFIG_PATH='/home/opencv-mds/tf_ssd/tod/train_models/research/object_detection/samples/configs/ssd_inception_v2_coco.config'

$ MODEL_DIR='/home/opencv-mds/tf_ssd/tod/save_models/coco_test'

*vBox 의 경우 저장공간도 부족 !!

$ NUM_TRAIN_STEPS=20

$ NUM_EVAL_STEPS=2


$ SAMPLE_1_OF_N_EVAL_EXAMPLES=1

====================
-----------------트레이닝 중에는 터미널을 빠져나가면 안됨 !!!

vBox 는 gpu 가 없기 때문에 다음의 변수 지정이 필요하다고 함

export TF_CPP_MIN_LOG_LEVEL=2

$ python object_detection/model_main.py \
--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
--model_dir=${MODEL_DIR} \
--num_train_steps=${NUM_TRAIN_STEPS} \
--num_eval_steps=${NUM_EVAL_STEPS} \
--checkpoint_dir=${PRE_TRAIN} \
--sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
--num_clones=1 \
--ps_tasks=1


1시에 시작
10분이면 end
-

=============================

<<4 Export pb>>

$ cd ~/tf_ssd/tod/export_models/research

$ export PYTHONPATH=$PYTHONPATH:/home/opencv-mds/tf_ssd/tod/export_models/research:/home/opencv-mds/tf_ssd/tod/export_models/research/slim

$ protoc object_detection/protos/*.proto --python_out=.

$ python object_detection/builders/model_builder_test.py

cp ~/tf_ssd/tod/train_models/research/object_detection/samples/configs/ssd_inception_v2_coco.config ./object_detection/samples/configs/

$ vim ./object_detection/samples/configs/ssd_inception_v2_coco.config

line 101: override 부분 주석

$ INPUT_TYPE=image_tensor

-----------------------------
$ PIPELINE_CONFIG_PATH='/home/opencv-mds/tf_ssd/tod/export_models/research/object_detection/samples/configs/ssd_inception_v2_coco.config'
ls /home/opencv-mds/tf_ssd/tod/save_models/coco_test/

$ TRAINED_CKPT_PREFIX='/home/opencv-mds/tf_ssd/tod/save_models/coco_test/model.ckpt-20'

mkdir ~/tf_ssd/tod/pbfiles

========================
$ EXPORT_DIR='/home/opencv-mds/tf_ssd/tod/pbfiles'
$ python object_detection/export_inference_graph.py \
--input_type=${INPUT_TYPE} \
--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
--trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
--output_directory=${EXPORT_DIR}

$ ls ~/tf_ssd/tod/pbfiles/

