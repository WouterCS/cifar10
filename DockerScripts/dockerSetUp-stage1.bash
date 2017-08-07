NAME='tfGPU'
IMAGENAME='tensorflow/tensorflow:nightly-devel-gpu' # 'mydockerimage' #
PROJECTPATH="$(dirname "$(dirname "${BASH_SOURCE[0]}")")" #'/home/wouter/Documents/DockerMap/Code'

git pull

docker rm $(docker stop $(docker ps -aq --no-trunc))
nvidia-docker run -itd -p 8888:8888 -p 6006:6006 --name $NAME -v $PROJECTPATH/notebooks:/notebooks -v $PROJECTPATH/RFNN:/usr/local/lib/python2.7/dist-packages/RFNN $IMAGENAME /bin/bash

#git clone https://github.com/tensorflow/models.git

docker cp $PROJECTPATH/models/ $NAME:/models-2
docker cp $PROJECTPATH/DockerScripts/DockerCreation $NAME:/scripts
docker start $NAME 
docker exec -it $NAME /bin/bash /scripts/ondockerSetUp-stage1.bash

docker commit $NAME 'cifar-10-example'

# action to be performed on the docker container:
docker exec -it $NAME /bin/bash cd /models-2/cifar10
docker exec -it $NAME /bin/bash apt-get update
docker exec -it $NAME /bin/bash apt-get install tk-dev python-tk 
docker exec -it $NAME /bin/bash python -c "import  cifar10 as model; model.maybe_download_and_extract()"

# calliong the next script:
bash $PROJECTPATH/DockerScripts/dockerSetUp-stage2.bash