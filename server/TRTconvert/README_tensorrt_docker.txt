
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

 convert-to-uff --input-file /iitp_ws/tod0/pbfiles/frozen_inference_graph.pb -O NMS -p ./config.py

===> UFF Output written to /iitp_ws/tod0/pbfiles/frozen_inference_graph.uff
 
frozen_inference_graph.uff 가 생성되면 /workspace/tensorrt/data/ssd 폴더로 이동시킨 후 

frozen_inference_graph.uff -> sample_ssd_relu6.uff 로 변경한다.(mv)
# mv frozen_inference_graph.uff sample_ssd_relu6.uff

cp sample_ssd_relu6.uff /workspace/tensorrt/data/ssd/

cd /workspace/tensorrt/samples/
샘플 빌드
소스 49라인 수정 : threshold 값을 0.25 정도로 수정한다.  
make

cd bin/
./sampleUffSSD







