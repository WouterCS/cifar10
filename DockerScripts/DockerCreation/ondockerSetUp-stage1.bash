#apt-get update
#apt-get -y install python-tk


cd /models-2/cifar10

apt-get update
apt-get install tk-dev python-tk 
python -c "import  cifar10 as model; model.maybe_download_and_extract()"