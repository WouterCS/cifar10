from __future__ import print_function
import os.path
import math
import numpy as np
import tensorflow as tf
import math

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
            self.convNonLin = 'identity' # 'relu'   'powMagnitude'   'funMagnitude'
            self.convFunMagnitude = tf.pow
            self.convConstantMagnitude = 1.75
            self.poolingFun = 'max-pool' # 'average-pool'
            self.max_steps = 1000000
            self.steps_done_at_start = 0
            self.eval_frequency = 1000
            self.input_shuffle_seed = 0 #None
            
            self.non_linearity = {'FC': {'type_of_nonlin': 'identity',
                                         'apply_const_function': tf.pow,
                                         'const': 1.90},
                                  'conv': {'type_of_nonlin': 'identity',
                                           'apply_const_function': tf.pow,
                                           'const': 1.90}}
            
            self.INITIAL_LEARNING_RATE = 0.1
            self.current_lr = self.INITIAL_LEARNING_RATE
            self.FIXED_LR = True
            self.FIXED_LR_EVAL_CYCLES = 50
            self.MIN_LEARNING_RATE = 0.005
            self.LR_MULTIPLIER = 0.95
    
    hyperParam = hyperParameters()
    hyperParam.poolingFun = 'average-pool'
    hyperParam.INITIAL_LEARNING_RATE = 0.1
    hyperParam.FIXED_LR = True
    hyperParam.max_steps = 10000
    hyperParam.steps_done_at_start = 0
    
    hyperParam.non_linearity['conv']['type_of_nonlin'] = 'funMagnitude'
    
    addConsts = [1, 0.1, 0.5, 0.01]
    multConsts = [2, 0.5, 0.9, 1.01]
    if runNum % 2 == 0:
        hyperParam.non_linearity['conv']['apply_const_function'] = tf.add
        hyperParam.non_linearity['conv']['const'] = addConsts[runNum / 2] #np.random.random(1)[0] * 2
    else:
        hyperParam.non_linearity['conv']['apply_const_function'] = tf.multiply
        hyperParam.non_linearity['conv']['const'] = multConsts[runNum / 2] # 0.5 + np.random.random(1)[0]
    

    createReadMe(hyperParam)
    return hyperParam
    
def createReadMe(hyperParam):

    print('Directory is: %s' % hyperParam.directory + '/README.txt')
    with open(hyperParam.directory + '/README.txt', 'wb') as f:
        print('start making readme')
        print('Dataset: %s' % hyperParam.datasetname, file = f)
        
        if hyperParam.non_linearity['FC']['type_of_nonlin'] == 'powMagnitude':
            print('FC non-linearity function: %s, with constant: %f' % (hyperParam.non_linearity['FC']['apply_const_function'], hyperParam.non_linearity['FC']['const']), file = f)
        else:
            print('FC non-linearity: %s' % hyperParam.non_linearity['FC']['type_of_nonlin'], file = f)
            
        if hyperParam.non_linearity['conv']['type_of_nonlin'] == 'funMagnitude':
            print('Conv non-linearity function: %s, with constant: %f' % (hyperParam.non_linearity['conv']['apply_const_function'], hyperParam.non_linearity['conv']['const']), file = f)
        else:
            print('Conv non-linearity: %s' % hyperParam.non_linearity['conv']['type_of_nonlin'], file = f)
            
        print('Pooling function is: %s' % hyperParam.poolingFun, file = f)
        print('Maximum number of steps is: %d' % hyperParam.max_steps, file = f)
        print('Evaluation every %d steps.' % hyperParam.eval_frequency, file = f)
        print('Learning-rate is: %f' % hyperParam.INITIAL_LEARNING_RATE, file = f)
        print('finished making readme')