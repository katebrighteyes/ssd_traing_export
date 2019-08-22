
sudo NV_GPU=0 nvidia-docker run --name IITPTRT0 -it -d --net=host \
 -v "/drv3/iitp1/share:/iitp_ws" \
 nvcr.io/nvidia/tensorrt:19.01-py3

sudo docker restart IITPTRT0

* IITPTRT0 라는 컨테이너에 들어간다.
sudo docker exec -it IITPTRT0 /bin/bash

*run this sh srcript as soon as container start.
*최초 한번 실행
/opt/tensorrt/python/python_setup.sh

2.  pb 를 UFF 컨버팅
도커 안의 tensorrt/samples/sampleUffSSD 에서 pb 를 컨버팅

cd /workspace/tensorrt/samples/sampleUffSSD

cp /iitp_ws/config.py .

 convert-to-uff --input-file /data/frozen_inference_graph.pb -O NMS -p /data/config.py

===> UFF Output written to /data/frozen_inference_graph.uff
 
frozen_inference_graph.uff 가 생성되면 tensorrt/data/ssd 폴더로 이동시킨 후 sample_ssd_relu6.uff 로 변경한다.

/data# mv frozen_inference_graph.uff sample_ssd_relu6.uff
cp sample_ssd_relu6.uff /workspace/tensorrt/data/ssd/

cd /workspace/tensorrt/samples/
make

cd bin/bash







