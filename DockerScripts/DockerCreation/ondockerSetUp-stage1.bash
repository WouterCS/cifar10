#apt-get update
#apt-get -y install python-tk


cd /models/cifar10

python -c "import  cifar10 as model; model.maybe_download_and_extract()" 2>&1  | tee /results/log.txt