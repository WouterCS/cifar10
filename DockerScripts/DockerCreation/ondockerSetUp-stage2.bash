mkdir -p /results

cd /models/cifar10

#python -c "from tensorflow.contrib.layers.python.layers.embedding_ops import *"
#cat /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/layers/python/layers/embedding_ops.py

for i in {7..8}
do
    numRun=i
    DIRECTORY='/results/run-'$numRun
    mkdir -p $DIRECTORY
    echo $"${DIRECTORY}/log.txt"
    python -c "import cifar10_params; hyperParam = cifar10_params.main(${numRun},'${DIRECTORY}'); import cifar10_train; cifar10_train.main(hyperParam, '${DIRECTORY}')" 2>&1  | tee "${DIRECTORY}/log.txt"
done 