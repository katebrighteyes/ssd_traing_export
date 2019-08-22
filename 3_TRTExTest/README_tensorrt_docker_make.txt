1. 도커 설치
이걸 nvcr.io/nvidia/tensorrt:19.01-py3 도커 이미지를 pull

docker pull nvcr.io/nvidia/tensorrt:19.03-py3

alias 작성 요망 

2. ~/.bashrc 에 Container alias 추가
alias TRT5_START='nvidia-docker run -it -d --name TENSORRT5 --net=host -v /home/drivepx/tensorrt:/data nvcr.io/nvidia/tensorrt:19.01-py3'
alias TRT5_ATTACH='docker exec -it TENSORRT5 /bin/bash'
alias TRT5_RESTART='docker restart TENSORRT5'

source ~/.bashrc

TRT5_START
TRT5_ATTACH
TRT5_RESTART

