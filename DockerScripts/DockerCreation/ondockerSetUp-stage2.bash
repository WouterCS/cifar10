mkdir -p /results

python -c "import  models/tutorials/image/cifar10/cifar10_train as model; model.train()" 2>&1  | tee /results/log.txt
#python -c "import RFNN.trainnonlin.mod_training as train; train.run()" 2>&1  | tee /results/log.txt