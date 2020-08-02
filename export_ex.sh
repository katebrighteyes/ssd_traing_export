INPUT_TYPE=image_tensor

PIPELINE_CONFIG_PATH='/tf_ssd/tod/export_models/research/object_detection/samples/configs/ssd_inception_v2_coco.config'

TRAINED_CKPT_PREFIX='/tf_ssd/tod/save_models/coco_test/model.ckpt-20'

EXPORT_DIR='/tf_ssd/tod/pbfiles'

python object_detection/export_inference_graph.py --input_type=${INPUT_TYPE} --pipeline_config_path=${PIPELINE_CONFIG_PATH} --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} --output_directory=${EXPORT_DIR}
