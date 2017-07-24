mkdir -p /results

cd /models/cifar10

DIRECTORY='/results/run-1'
mkdir -p $DIRECTORY
python -c "import params; hyperParam = params.main(1); import cifar10_train; cifar10_train.main(hyperParam, '${DIRECTORY}')" 2>&1  | tee "${DIRECTORY}/log.txt"
#cat /models/cifar10/cifar10_train.py 2>&1  | tee /results/log.txt

#python -c "import RFNN.trainnonlin.mod_training as train; train.run()" 2>&1  | tee /results/log.txt