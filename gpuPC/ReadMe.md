[Folder Tree]

/ssd_ws

=

/tf_ssd/share

     dataset/cocodata/tfrecords/
     
<tfssd>
sudo NV_GPU=0 nvidia-docker run --name TFSSD00 -it -d --net=host \
 -v "/drv3/tf_ssd/share:/ssd_ws" \
 tensorflow/tensorflow:1.14.0-gpu-py3

sudo docker exec -it TFSSD /bin/bash

sudo docker restart TFSSD
