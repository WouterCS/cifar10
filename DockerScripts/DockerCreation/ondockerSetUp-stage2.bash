mkdir -p /results

cd /models/cifar10

DIRECTORY='/results/run-2'
mkdir -p $DIRECTORY
python -c "import cifar10_params; hyperParam = cifar10_params.main(2); import cifar10_train; cifar10_train.main(hyperParam, '${DIRECTORY}')" 2>&1  | tee "${DIRECTORY}/log.txt"

DIRECTORY='/results/run-3'
mkdir -p $DIRECTORY
python -c "import cifar10_params; hyperParam = cifar10_params.main(3); import cifar10_train; cifar10_train.main(hyperParam, '${DIRECTORY}')" 2>&1  | tee "${DIRECTORY}/log.txt"

DIRECTORY='/results/run-4'
mkdir -p $DIRECTORY
python -c "import cifar10_params; hyperParam = cifar10_params.main(4); import cifar10_train; cifar10_train.main(hyperParam, '${DIRECTORY}')" 2>&1  | tee "${DIRECTORY}/log.txt"
