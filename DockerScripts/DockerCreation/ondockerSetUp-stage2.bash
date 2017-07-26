mkdir -p /results

cd /models/cifar10

python -c "from tensorflow.contrib.layers.python.layers.embedding_ops import *"
ls /usr/local/lib/python2.7/dist-packages/


DIRECTORY='/results/run-4'
mkdir -p $DIRECTORY
python -c "import cifar10_params; hyperParam = cifar10_params.main(1,'${DIRECTORY}'); import cifar10_train; cifar10_train.main(hyperParam, '${DIRECTORY}')" 2>&1  | tee "${DIRECTORY}/log.txt"
