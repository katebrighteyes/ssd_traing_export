apt-get install git vim

tensorflow/tensorflow:1.12.0-gpu-py3

git checkout 68c3c65596b8fc624be15aef6eac3dc8952cbf23


# docker shared folder : gpu server folder position

/ssd_ws

=

/drv3/tf_ssd/

             dataset/cocodata/tfrecords/
             
             tod0/
             
             tod1/
             
             tod2/

             tod3/

             tod4/

             tod5/


# execute docker terminal

sudo docker exec -it TFSSD0 /bin/bash

# restart docker container

sudo docker restart TFSSD0
