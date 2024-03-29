# guide for PC (with GPU)

ssd_traing_export
#######################################################

This is guide for 10 minutes training on No GPU PC.

So it can NEVER be perfect training !!

#######################################################

# docker image pull

sudo nvidia-docker pull tensorflow/tensorflow:1.15.3-gpu-py3-jupyter


## make folder for docker container

cd /

sudo mkdir tf_ssd

cd tf_ssd

sudo chmod 777 .

#########################################

# docker container

sudo nvidia-docker run --name TFSSD -it -d --net=host \
 -v "/tf_ssd:/rf_ssd" \
 tensorflow/tensorflow:1.15.3-gpu-py3-jupyter

sudo docker exec -it TFSSD /bin/bash

sudo docker restart TFSSD

# 2-0 install package

















# 2-1 Tensorflow TEST

mkdir mnist

cd mnist

mkdir temp

vim mnist_cnn.py

아래 링크를 보고 작성

https://github.com/katebrighteyes/PythonDL_Collection/blob/master/mnist/mnist_cnn.py

저장한 후 권한 변경

chmod +x mnist_cnn.py

python mnist_cnn.py

# 2-2 Tensorflow slim TEST

$ cd /tf_ssd/

mkdir tod

cd tod


$ git clone https://github.com/tensorflow/models.git

$ mv models train_models

cd train_models

git checkout 5ed215b2ae0fd9650d1650953afcffdd23bb28f6

cd research/slim


* train_image_classifier.py 592 line 에 다음과 같이 수정

-------------------------

vim train_image_classifier.py

    session_config = tf.ConfigProto(allow_soft_placement=True)

    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None,
        session_config = session_config,
        )

-----------------------------------
cd /tf_ssd/tod

/tf_ssd/tod$ mkdir googlenet

cd googlenet

wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz

tar xzf inception_v1_2016_08_28.tar.gz

* 데이터 다운로드

cd  /tf_ssd/tod/train_models/research/slim

python download_and_convert_data.py --dataset_name=flowers --dataset_dir=../../../flowers
    

* 훈련 inceptionnet classification 

$ python train_image_classifier.py \
    --train_dir=/tf_ssd/tod/trained \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=/tf_ssd/tod/flowers \
    --model_name=inception_v1 \
    --max_number_of_steps=500 \
    --checkpoint_path=/tf_ssd/tod/googlenet/inception_v1.ckpt \
    --checkpoint_exclude_scopes=InceptionV1/Logits \
    --trainable_scopes=InceptionV1/Logits \
    --batch_size=16 \
    --learning_rate=0.01 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=100 \
    --log_every_n_steps_secs=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004
    
  * ---> 한 줄로 표현
 $ python train_image_classifier.py --train_dir=/tf_ssd/tod/trained --dataset_name=flowers --dataset_split_name=train   --dataset_dir=/tf_ssd/tod/flowers --model_name=inception_v1 --max_number_of_steps=500  --checkpoint_path=/tf_ssd/tod/googlenet/inception_v1.ckpt --checkpoint_exclude_scopes=InceptionV1/Logits --trainable_scopes=InceptionV1/Logits --batch_size=16  --learning_rate=0.01  --learning_rate_decay_type=fixed  --save_interval_secs=100  --log_every_n_steps_secs=100  --optimizer=rmsprop --weight_decay=0.00004
    
* 평가

mkdir /tf_ssd/tod/trained

$ python eval_image_classifier.py \
    -–alsologtostderr \
    --checkpoint_path=/tf_ssd/tod/trained \
    --dataset_dir=/tf_ssd/tod/flowers \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --model_name=inception_v1

  * ---> 한 줄로 표현
 $ python eval_image_classifier.py -–alsologtostderr  --checkpoint_path=/tf_ssd/tod/trained --dataset_dir=/tf_ssd/tod/flowers  --dataset_name=flowers  --dataset_split_name=validation --model_name=inception_v1


# 3 SSD Object Detection

동일한 train_model 폴더에서 진행

# pycocotools, protocbuf install

-----pycocotools install -----

$ cd /tf_ssd/tod/train_models/research

$ export PYTHONPATH=$PYTHONPATH:/tf_ssd/tod/train_models/research:/tf_ssd/tod/train_models/research/slim

$ git clone https://github.com/cocodataset/cocoapi.git

$ cd cocoapi/PythonAPI

$ make

$ cp -r pycocotools /tf_ssd/tod/train_models/research/


-----protocbuf install -------

$ cd /tf_ssd/tod/train_models/research

$ curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip

$ unzip protoc-3.2.0-linux-x86_64.zip -d protoc3

$ sudo mv protoc3/bin/* /usr/local/bin/

$ sudo mv protoc3/include/* /usr/local/include/

$ protoc object_detection/protos/*.proto --python_out=.

$ python object_detection/builders/model_builder_test.py

------------Just you can see "OK" -> it is ok !!


# 3-2 tfrecord 준비

* ctl-alt-t to open onother terminal.

cd /tf_ssd/

scp ?? /tfrecord.zip ./

unzip tfrecord.zip

===========================================

# 3-3 train model modify

cd /tf_ssd/tod/train_models/research

$ vim /tf_ssd/tod/train_models/research/object_detection/samples/configs/ssd_inception_v2_coco.config

line: 151, 152 -> 주석(#) 처리

fine_tune_checkpoint: "/tf_ssd/pretrained/model.ckpt-6919"   
from_detection_checkpoint: true 

line: 170,184 -> path설정

해당 라인에 적혀있는 path의 tfrecord를 train하므로 우리데이터셋 경로로 바꿔주자.

170, 184: /tf_ssd/tfrecord/ 여기에 ms만 지우면됨

line: 172,186 -> mscoco_label_map.pbtxt 경로를 설정해줘야 한다.

172, 186: /tf_ssd/tod/train_models/research/object_detection/data/mscoco_label_map.pbtxt

++++++++++++++++++++++++++++++++++++++++++

수정된 config 예 : vBox에 coco_val.record 가 없어서 동일하게 coco_train.record 을 사용함.

- 훈련 데이터 관련 train_input_reader

169 train_input_reader: {

170 tf_record_input_reader {

171 input_path: "/tf_ssd/tfrecord/coco_train.record-?????-of-00100"

172 }

173 label_map_path: "/tf_ssd/tod/train_models/research/object_detection/data/mscoco_label_map.pbtxt"

174 }

- 평가 데이터 관련 eval_input_reader

183 eval_input_reader: {


184 tf_record_input_reader {

185 input_path: "/tf_ssd/tfrecord/coco_train.record-?????-of-00100"

186 }

187 label_map_path: "/tf_ssd/tod/train_models/research/object_detection/data/mscoco_label_map.pbtxt"

188 shuffle: false

189 num_readers: 1

190 }

메모리 여유가 없는 경우 136line 의 배치 사이즈를 4 나 16로 수정 필요 !! (vm의 메모리를 보고 결정0


# 3-4 train

이제 학습에 필요한 파라미터들을 설정해주고 실행하면 된다.

mkdir /tf_ssd/tod/save_models/

mkdir /tf_ssd/tod/save_models/coco_test

$ PIPELINE_CONFIG_PATH='/tf_ssd/tod/train_models/research/object_detection/samples/configs/ssd_inception_v2_coco.config'

$ MODEL_DIR='/tf_ssd/tod/save_models/coco_test'


$ NUM_TRAIN_STEPS=20

$ NUM_EVAL_STEPS=2

$ SAMPLE_1_OF_N_EVAL_EXAMPLES=1


==================== 

-----------------트레이닝 중에는 터미널을 빠져나가면 안됨 !!!

* gpu 가 없기 때문에 다음의 변수 지정이 필요하다고 함

export TF_CPP_MIN_LOG_LEVEL=2


$ python object_detection/model_main.py --pipeline_config_path=${PIPELINE_CONFIG_PATH} --model_dir=${MODEL_DIR} --num_train_steps=${NUM_TRAIN_STEPS} --num_eval_steps=${NUM_EVAL_STEPS} --checkpoint_dir=${PRE_TRAIN} --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES --num_clones=1 --ps_tasks=1


1시에 시작 10분이면 end

============================================

* 텐서 보드 보기

tensorboard --logdir=/tf_ssd/save_models/coco_test


============================================

# dectivate !!!!

$ deactivate

# Open onother terminal

$ cd /

$ cd tf_ssd

$ source ./venvssd/bin/activate

=================================================
# 4 Export pb

# 4-1 models for Export

$ cd tod

$ git clone https://github.com/tensorflow/models.git

$ mv models export_models

$ cd export_models

$ git checkout ae0a9409212d0072938fa60c9f85740bb89ced7e

Don't be afraid to see Error !

please check different branch* 

$ cd research

$ export PYTHONPATH=$PYTHONPATH:/tf_ssd/tod/export_models/research:/tf_ssd/tod/export_models/research/slim

$ protoc object_detection/protos/*.proto --python_out=.

$ python object_detection/builders/model_builder_test.py

cp /tf_ssd/tod/train_models/research/object_detection/samples/configs/ssd_inception_v2_coco.config ./object_detection/samples/configs/

$ vim ./object_detection/samples/configs/ssd_inception_v2_coco.config

line 101: override 부분 주석

$ INPUT_TYPE=image_tensor


$ PIPELINE_CONFIG_PATH='/tf_ssd/tod/export_models/research/object_detection/samples/configs/ssd_inception_v2_coco.config'

ls /tf_ssd/tod/save_models/coco_test/

$ TRAINED_CKPT_PREFIX='/tf_ssd/tod/save_models/coco_test/model.ckpt-10000'

mkdir /tf_ssd/tod/pbfiles

========================

$ EXPORT_DIR='/tf_ssd/tod/pbfiles'

$ python object_detection/export_inference_graph.py --input_type=${INPUT_TYPE} --pipeline_config_path=${PIPELINE_CONFIG_PATH} --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} --output_directory=${EXPORT_DIR}


$ ls /tf_ssd/tod/pbfiles/
