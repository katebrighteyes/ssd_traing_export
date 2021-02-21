***********************************************************************
실습 시에는 만들어진 것을 사용해주시고
현업에서는 한번 만들어서 계속 사용하면 됩니다.

sudo nvidia-docker run --name TFTRT -it -d --net=host \
 -v "/tf_ssd:/tf_ssd" \
 nvcr.io/nvidia/tensorrt:19.01-py3
 
 혹시 에러가 나면 
 sudo systemctl restart docker
 먼저 실행하고 다시 시도해주세요.
 
************************************************************************


* IITPTRT 라는 컨테이너를 시작한다.
sudo docker restart TFTRT

* IITPTRT 라는 컨테이너에 들어간다.
sudo docker exec -it TFTRT /bin/bash

***********************************************************************
*run this sh srcript as soon as container start.
*최초 한번 실행
* 이미 만들어진 컨테이너에서는 실행하지 말아주십시오.

/opt/tensorrt/python/python_setup.sh

혹시 grpcio 실행 중 진행이 멈춰져 있으면 Ctl-C 로 멈추고
아래와 같이 grpcio 를 설치해주세요.
pip install grpcio==1.11.0
그리고 나서 다시 /opt/tensorrt/python/python_setup.sh 실행해주세요.
************************************************************************

2.  pb 를 UFF 컨버팅
* 도커 안의 tensorrt/samples/sampleUffSSD 에서 pb 를 컨버팅

# cd /workspace/tensorrt/samples/sampleUffSSD

# convert-to-uff --input-file /tf_ssd/convert/frozen_inference_graph.pb -O NMS -p /tf_ssd/convert/config.py

* ===> UFF Output written to /tf_ssd/convert/frozen_inference_graph.uff
 
* frozen_inference_graph.uff 가 생성되면 /workspace/tensorrt/data/ssd 폴더로 이동시킨

# cp /tf_ssd/convert/frozen_inference_graph.uff /workspace/tensorrt/data/ssd/sample_ssd_relu6.uff

# ls /workspace/tensorrt/data/ssd

# cd /workspace/tensorrt/samples/sampleUffSSD

# vim sampleUffSSD.cpp

소스 49라인 수정 : threshold 값을 0.25 정도로 수정한다.  


샘플 빌드

# cd /workspace/tensorrt/samples/

# make

# cd ../bin/

* if there are ppm file, the you better remove them.

# rm *.ppm
# ./sample_uff_ssd

* and check the output files.
cp car-*.ppm /tf_ssd/ppm




