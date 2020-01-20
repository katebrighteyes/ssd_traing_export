# 4. delete previous trained files
 
rm -rf ~/tf_ssd/tod/save_models/coco_test/*

# 5. object detection setting
 
cd /home/opencv-mds/tf_ssd/tod/train_models/research

PIPELINE_CONFIG_PATH='/home/opencv-mds/tf_ssd/tod/train_models/research/object_detection/samples/configs/ssd_inception_v2_coco.config'

MODEL_DIR='/home/opencv-mds/tf_ssd/tod/save_models/coco_test'
NUM_TRAIN_STEPS=20
NUM_EVAL_STEPS=2
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
export TF_CPP_MIN_LOG_LEVEL=2

# 6. object detection start training

python object_detection/model_main.py --pipeline_config_path=${PIPELINE_CONFIG_PATH} --model_dir=${MODEL_DIR} --num_train_steps=${NUM_TRAIN_STEPS} --num_eval_steps=${NUM_EVAL_STEPS} --checkpoint_dir=${PRE_TRAIN} --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES --num_clones=1 --ps_tasks=1


