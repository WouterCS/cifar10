mkdir -p /results

cd /models/cifar10

python -c "import  cifar10_train as model; model.main('identity', 'identity');import cifar10_eval as eval; eval.main('identity', 'identity')" 2>&1  | tee /results/log.txt
python -c "import  cifar10_train as model; model.main('relu', 'identity');import cifar10_eval as eval; eval.main('relu', 'identity')" 2>&1  | tee /results/log.txt
#cat /models/cifar10/cifar10_train.py 2>&1  | tee /results/log.txt

#python -c "import RFNN.trainnonlin.mod_training as train; train.run()" 2>&1  | tee /results/log.txt