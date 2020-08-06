***********************************************************************
실습 시에는 만들어진 것을 사용해주시고
현업에서는 한번 만들어서 계속 사용하면 됩니다.

sudo nvidia-docker run --name TFTRT0 -it -d --net=host \
 -v "/tf_ssd/convert:/ssd_ws" \
 nvcr.io/nvidia/tensorrt:19.01-py3
 
 혹시 에러가 나면 
 sudo systemctl restart docker
 먼저 실행하고 다시 시도해주세요.
 
************************************************************************


* IITPTRT0 라는 컨테이너를 시작한다.
sudo docker restart TFTRT0

* IITPTRT0 라는 컨테이너에 들어간다.
sudo docker exec -it TFTRT0 /bin/bash

***********************************************************************
*run this sh srcript as soon as container start.
*최초 한번 실행
* 이미 만들어진 컨테이너에서는 실행하지 말아주십시오.

*pip install grpcio==1.11.0

/opt/tensorrt/python/python_setup.sh
************************************************************************

2.  pb 를 UFF 컨버팅
* 도커 안의 tensorrt/samples/sampleUffSSD 에서 pb 를 컨버팅

# cd /workspace/tensorrt/samples/sampleUffSSD

# convert-to-uff --input-file /ssd_ws/convert/frozen_inference_graph.pb convert-to-uff --input-file /ssd_ws/

* ===> UFF Output written to /ssd_ws/convert/frozen_inference_graph.uff
 
* frozen_inference_graph.uff 가 생성되면 /workspace/tensorrt/data/ssd 폴더로 이동시킨

# cp /ssd_ws/convert/frozen_inference_graph.uff /workspace/tensorrt/data/ssd/sample_ssd_relu6.uff

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
cp car-*.ppm /ssd_ws/




