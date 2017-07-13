NAME='tfGPU'
IMAGENAME='cifar-10-example'
LOCALPATH='/home/uijenswr/Documents/tempResults'
DROPBOXPATH='/home/uijenswr/Dropbox/thesis' #'/home/wouter/Documents/localResults' #


PROJECTPATH="$(dirname "$(dirname "${BASH_SOURCE[0]}")")" 
docker rm $(docker stop $(docker ps -aq --no-trunc))
nvidia-docker run -itd -p 8888:8888 -p 6006:6006 --name $NAME -v $DROPBOXPATH/results:/results $IMAGENAME 

cd $PROJECTPATH
git pull
docker cp $PROJECTPATH/models/ $NAME:/models
docker cp $PROJECTPATH/DockerScripts/DockerCreation $NAME:/scripts
docker start $NAME 
docker exec -it $NAME /bin/bash /scripts/ondockerSetUp-stage2.bash
#mv $LOCALPATH              $DROPBOXPATH

