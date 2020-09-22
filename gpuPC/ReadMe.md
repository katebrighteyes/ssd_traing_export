[Folder Tree]

/ssd_ws

=

/tf_ssd/shared

     dataset/cocodata/tfrecords/
     
<tfssd>
     
sudo nvidia-docker run --name TFSSD -it -d --net=host \
     
 -v "/tf_ssd/shared:/ssd_ws" \ 
 tensorflow/tensorflow:1.14.0-gpu-py3

sudo docker exec -it TFSSD /bin/bash

sudo docker restart TFSSD
