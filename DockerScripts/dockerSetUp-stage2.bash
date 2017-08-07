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
docker cp $PROJECTPATH/DockerScripts/DockerCreation $NAME:/scripts2
docker start $NAME 
#docker exec -it $NAME /bin/bash /scripts2/ondockerSetUp-stage2.bash


docker exec -it $NAME mkdir -p /results
docker exec -it $NAME cd /models/cifar10

for i in {1..1000}
do
    numRun=$i
    DIRECTORY='/results/run-'$numRun
    docker exec -it $NAME mkdir -p $DIRECTORY
    docker exec -it $NAME python -c "import cifar10_params; hyperParam = cifar10_params.main(${numRun},'${DIRECTORY}'); import cifar10_train; cifar10_train.main(hyperParam, '${DIRECTORY}')" 2>&1  | tee "${DIRECTORY}/log.txt"
done 

