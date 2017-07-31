from __future__ import print_function
import os.path
import math
import numpy as np

def main(runNum, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    class hyperParameters:
        def __init__(self):
            self.directory = directory
            self.train_dir = directory + '/cifar10_train'
            self.datasetname = 'CIFAR-10'
            self.FCnonLin = 'identity' # 'relu'   'powMagnitude'
            self.FCnonLinMag = 0.9
            self.convNonLin = 'identity' # 'relu'   'powMagnitude'
            self.convNonLinMag = 0.9
            self.poolingFun = 'max-pool' # 'average-pool'
            self.max_steps = 1000000
            self.eval_frequency = 1000
            self.input_shuffle_seed = 0 #None
            
            self.INITIAL_LEARNING_RATE = 0.1
            self.current_lr = self.INITIAL_LEARNING_RATE
            self.FIXED_LR = True
            self.FIXED_LR_EVAL_CYCLES = 50
            self.MIN_LEARNING_RATE = 0.005
            self.LR_MULTIPLIER = 0.9
    
    hyperParam = hyperParameters()
    hyperParam.poolingFun = 'average-pool'
    hyperParam.INITIAL_LEARNING_RATE = 0.1
        
    hyperParam.max_steps = 30000
    hyperParam.convNonLin = 'powMagnitude'
    hyperParam.convNonLinMag = 2 + 0.2 * runNum
    
    createReadMe(hyperParam)
    return hyperParam
    
def createReadMe(hyperParam):

    print('Directory is: %s' % hyperParam.directory + '/README.txt')
    with open(hyperParam.directory + '/README.txt', 'wb') as f:
        print('start making readme')
        print('Dataset: %s' % hyperParam.datasetname, file = f)
        if hyperParam.FCnonLin == 'powMagnitude':
            print('FC non-linearity: %s, with power: %f' % (hyperParam.FCnonLin, hyperParam.FCnonLinMag), file = f)
        else:
            print('FC non-linearity: %s' % hyperParam.FCnonLin, file = f)
        if hyperParam.convNonLin == 'powMagnitude':
            print('Conv non-linearity: %s, with power: %f' % (hyperParam.convNonLin, hyperParam.convNonLinMag), file = f)
        else:
            print('Conv non-linearity: %s' % hyperParam.convNonLin, file = f)
        print('Pooling function is: %s' % hyperParam.poolingFun, file = f)
        print('Maximum number of steps is: %d' % hyperParam.max_steps, file = f)
        print('Evaluation every %d steps.' % hyperParam.eval_frequency, file = f)
        print('Learning-rate is: %f' % hyperParam.INITIAL_LEARNING_RATE, file = f)
        print('finished making readme')