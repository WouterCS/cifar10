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
            self.poolingFun = 'max-pool' # 'average-pool'
            self.max_steps = 1000000
            self.steps_done_at_start = 0
            self.eval_frequency = 1000
            self.input_shuffle_seed = 0 #None
            
            self.non_linearity = {'FC': {'type_of_nonlin': 'identity', # 'relu'   'powMagnitude'   'funMagnitude'  'funAngle' 'expFFT'   'funMagnitudeSecFunAngle'   'applyToCartOfComplex'
                                         'apply_const_function': tf.pow,
                                         'const': 1.90,
                                         'normalizeAngle': False,
                                         'anglePositiveValued': False,
                                         'secondary_const_fun': tf.multiply,
                                         'secondary_const': 0.25},
                                  'conv': {'type_of_nonlin': 'identity',
                                           'apply_const_function': tf.pow,
                                           'const': 1.90,
                                           'normalizeAngle': False,
                                           'anglePositiveValued': False,
                                           'secondary_const_fun': tf.multiply,
                                           'secondary_const': 0.25}}
            
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
    
    
    constsToTest = [(20,0), (20,20), (0,5), (0,20), (10,10), (5,5)] 
    
    const1, const2 = constsToTest[runNum]
    print('const1: %f, const2: %f' % (const1, const2))
    hyperParam.non_linearity['conv']['type_of_nonlin'] = 'applyToCartOfComplex'
    hyperParam.non_linearity['conv']['apply_const_function'] = tf.add
    hyperParam.non_linearity['conv']['const'] = const1
    hyperParam.non_linearity['conv']['secondary_const_fun'] = tf.add
    hyperParam.non_linearity['conv']['secondary_const'] = const2
    hyperParam.non_linearity['conv']['normalizeAngle'] = False
    hyperParam.non_linearity['conv']['anglePositiveValued'] = False
    
    
    createReadMe(hyperParam)
    return hyperParam
    
    # totalNumOfExperiments = 
    
    # if runNum % totalNumOfExperiments == :
        # hyperParam.non_linearity['conv']['type_of_nonlin'] = 'applyToCartOfComplex'
        # hyperParam.non_linearity['conv']['apply_const_function'] = tf.add
        # hyperParam.non_linearity['conv']['const'] = 
        # hyperParam.non_linearity['conv']['secondary_const_fun'] = 
        # hyperParam.non_linearity['conv']['secondary_const'] = 
        # hyperParam.non_linearity['conv']['normalizeAngle'] = 
        # hyperParam.non_linearity['conv']['anglePositiveValued'] = 
    
    # if runNum % totalNumOfExperiments == :
        # hyperParam.non_linearity['conv']['type_of_nonlin'] = 'applyToCartOfComplex'
        # hyperParam.non_linearity['conv']['apply_const_function'] = 
        # hyperParam.non_linearity['conv']['const'] = 
        # hyperParam.non_linearity['conv']['secondary_const_fun'] = 
        # hyperParam.non_linearity['conv']['secondary_const'] = 
        # hyperParam.non_linearity['conv']['normalizeAngle'] = 
        # hyperParam.non_linearity['conv']['anglePositiveValued'] = 
    
    # createReadMe(hyperParam)
    # return hyperParam
    
def createReadMe(hyperParam):

    print('Directory is: %s' % hyperParam.directory + '/README.txt')
    with open(hyperParam.directory + '/README.txt', 'wb') as f:
        print('start making readme')
        print('Dataset: %s' % hyperParam.datasetname, file = f)
        for layer in ['FC', 'conv']:
            print('For the %s layer:' % layer, file = f)
            print('  The non linearity function is: %s' % hyperParam.non_linearity[layer]['type_of_nonlin'], file = f)
            print('  Where applicable, the const function is: %s' % hyperParam.non_linearity[layer]['apply_const_function'], file = f)
            print('  Where applicable, the const is: %s' % hyperParam.non_linearity[layer]['const'], file = f)
            print('  Where applicable, the second const function is: %s' % hyperParam.non_linearity[layer]['secondary_const_fun'], file = f)
            print('  Where applicable, the secondary const is: %s' % hyperParam.non_linearity[layer]['secondary_const'], file = f)
            print('  do we normalize the angle? %s' % str(hyperParam.non_linearity[layer]['normalizeAngle']), file = f)
            
        print('Pooling function is: %s' % hyperParam.poolingFun, file = f)
        print('Maximum number of steps is: %d' % hyperParam.max_steps, file = f)
        print('Evaluation every %d steps.' % hyperParam.eval_frequency, file = f)
        print('Learning-rate is: %f' % hyperParam.INITIAL_LEARNING_RATE, file = f)
        print('finished making readme')