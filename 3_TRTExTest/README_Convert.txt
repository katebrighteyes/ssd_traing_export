TENSORRT 5.0


Start image and grnerate container

run this sh srcript as soon as container start.

/opt/tensorrt/python/python_setup.sh

2.  pb 를 UFF 컨버팅
도커 안의 tensorrt/samples/sampleUffSSD 에서 pb 를 컨버팅
 convert-to-uff --input-file /data/frozen_inference_graph.pb -O NMS -p /data/config.py

===> UFF Output written to /data/frozen_inference_graph.uff
 
frozen_inference_graph.uff 가 생성되면 tensorrt/data/ssd 폴더로 이동시킨 후 sample_ssd_relu6.uff 로 변경한다.

/data# mv frozen_inference_graph.uff sample_ssd_relu6.uff
cp sample_ssd_relu6.uff /workspace/tensorrt/data/ssd/

sampleUffSSD 예제를 실행시키면 ...

--결과---
Begin parsing model...
End parsing model...
Begin building engine...
End building engine...
 Num batches  2
 Data Size  540000
*** deserializing
Time taken for inference is 7.4731 ms.
 KeepCount 100


끝~~!!!!
