mkdir -p /results

cd /models/cifar10

python -c "convNonLin = 'identity'; FCnonLin = 'identity'; print('convNonLin: %s, FCnonLin: %s' % (convNonLin, FCnonLin)); import  cifar10_train as model; model.main(convNonLin, FCnonLin);import cifar10_eval as eval; eval.main(convNonLin, FCnonLin)" 2>&1  | tee /results/log.txt
python -c "convNonLin = 'relu'; FCnonLin = 'identity'; print('convNonLin: %s, FCnonLin: %s' % (convNonLin, FCnonLin)); import  cifar10_train as model; model.main(convNonLin, FCnonLin);import cifar10_eval as eval; eval.main(convNonLin, FCnonLin)" 2>&1  | tee /results/log2.txt
python -c "convNonLin = 'identity'; FCnonLin = 'relu'; print('convNonLin: %s, FCnonLin: %s' % (convNonLin, FCnonLin)); import  cifar10_train as model; model.main(convNonLin, FCnonLin);import cifar10_eval as eval; eval.main(convNonLin, FCnonLin)" 2>&1  | tee /results/log3.txt
python -c "convNonLin = 'relu'; FCnonLin = 'relu'; print('convNonLin: %s, FCnonLin: %s' % (convNonLin, FCnonLin)); import  cifar10_train as model; model.main(convNonLin, FCnonLin);import cifar10_eval as eval; eval.main(convNonLin, FCnonLin)" 2>&1  | tee /results/log4.txt
#cat /models/cifar10/cifar10_train.py 2>&1  | tee /results/log.txt

#python -c "import RFNN.trainnonlin.mod_training as train; train.run()" 2>&1  | tee /results/log.txt