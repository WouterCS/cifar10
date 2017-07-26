mkdir -p /results

cd /models/cifar10

#python -c "from tensorflow.contrib.layers.python.layers.embedding_ops import *"
#cat /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/layers/python/layers/embedding_ops.py

num = '4'
DIRECTORY='/results/run-'$num
mkdir -p $DIRECTORY
python -c "import cifar10_params; hyperParam = cifar10_params.main('$num','${DIRECTORY}'); import cifar10_train; cifar10_train.main(hyperParam, '${DIRECTORY}')" 2>&1  | tee "${DIRECTORY}/log.txt"
