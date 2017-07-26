mkdir -p /results

cd /models/cifar10

python -c "from tensorflow.contrib.layers import *" #.python.layers.embedding_ops

DIRECTORY='/results/run-4'
mkdir -p $DIRECTORY
python -c "import cifar10_params; hyperParam = cifar10_params.main(1,'${DIRECTORY}'); import cifar10_train; cifar10_train.main(hyperParam, '${DIRECTORY}')" 2>&1  | tee "${DIRECTORY}/log.txt"
