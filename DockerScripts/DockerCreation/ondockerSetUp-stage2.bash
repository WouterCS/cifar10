mkdir -p /results
cd /models/cifar10

echo '******************'
echo '* start from 10 *'
echo '******************'

for i in {10..200}
do
    numRun=$i
    DIRECTORY='/results/run-'$numRun
    mkdir -p $DIRECTORY
    python -c "import cifar10_params; hyperParam = cifar10_params.main(${numRun},'${DIRECTORY}'); import cifar10_train; cifar10_train.main(hyperParam, '${DIRECTORY}')" 2>&1  | tee "${DIRECTORY}/log.txt"
    sleep 2s
done 