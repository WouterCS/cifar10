from __future__ import print_function
import os.path
import math
import numpy as np
import tensorflow as tf
import math

import cifar10

def main(runNum, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    class hyperParameters:
        def __init__(self):
            self.directory = directory
            self.train_dir = directory + '/cifar10_train'
            self.datasetname = 'CIFAR-10'
            self.poolingFun = 'max-pool' # 'average-pool'   'max-pool'    'stride-pool'
            self.pool_strides = [1,2,2,1]
            self.max_steps = 1000000
            self.steps_done_at_start = 0
            self.eval_frequency = 1000
            self.input_shuffle_seed = 0 #None
            self.use_trainable_const = False
            
            self.non_linearity = {'FC': {'type_of_nonlin': 'identity', # 'relu'   'funMagnitude'  'funAngle' 'expFFT'   'funMagnitudeSecFunAngle'   'applyToCartOfComplex'   'applyToRealOfComplex'  'complexReLU'    'complexELU'  'full_taylor'
                                         'apply_const_function': tf.pow,
                                         'const': 1.90,
                                         'normalizeAngle': False,
                                         'anglePositiveValued': False,
                                         'secondary_const_fun': tf.multiply,
                                         'secondary_const': 0.25},
                                  'conv': {'type_of_nonlin': 'identity',
                                           'number_of_learned_weights': 1,
                                           'apply_const_function': tf.pow,
                                           'const': 1.48,
                                           'normalizeAngle': False,
                                           'anglePositiveValued': False,
                                           'secondary_const_fun': tf.multiply,
                                           'secondary_const': 0.25,
                                           'clip_min': 1e-4,
                                           'clip_max': 3}}
            self.INITIAL_LEARNING_RATE = 0.1
            self.current_lr = self.INITIAL_LEARNING_RATE
            self.FIXED_LR = True
            self.FIXED_LR_EVAL_CYCLES = 15
            self.MIN_LEARNING_RATE = 0.005
            self.LR_MULTIPLIER = 0.95
    
    hyperParam = hyperParameters()
    
    hyperParam.current_lr = 0.01
    hyperParam.FIXED_LR = True
    hyperParam.max_steps = 230000
    hyperParam.steps_done_at_start = 200000
    
    hyperParam.non_linearity['conv']['clip_min'] = -50
    hyperParam.non_linearity['conv']['clip_max'] = 50
    
    hyperParam.non_linearity['conv']['const'] = 1.48
    hyperParam.non_linearity['conv']['number_of_learned_weights'] = 2
    tests = {0: {'FC_non_lin': 'identity', 'conv_non_lin': 'full_taylor', 'pooling_function': 'average-pool', 'conv_const': 1.5 , 'learn_const': True},}
    # 0: {'FC_non_lin': 'identity', 'conv_non_lin': 'complexELU'  , 'pooling_function': 'average-pool', 'conv_const': -15 , 'learn_const': True},
             # 1: {'FC_non_lin': 'identity', 'conv_non_lin': 'complexReLU' , 'pooling_function': 'average-pool', 'conv_const': -15 , 'learn_const': True},
             # 2: {'FC_non_lin': 'relu'    , 'conv_non_lin': 'funMagnitude', 'pooling_function': 'stride-pool' , 'conv_const': 1.48, 'learn_const': False},
             # 3: {'FC_non_lin': 'identity', 'conv_non_lin': 'funMagnitude', 'pooling_function': 'max-pool'    , 'conv_const': 1.50, 'learn_const': True},
             # 4: {'FC_non_lin': 'relu'    , 'conv_non_lin': 'funMagnitude', 'pooling_function': 'max-pool'    , 'conv_const': 1.50, 'learn_const': True},
             # 5: {'FC_non_lin': 'identity', 'conv_non_lin': 'funMagnitude', 'pooling_function': 'average-pool', 'conv_const': 1.50, 'learn_const': True},
             # 6: {'FC_non_lin': 'relu'    , 'conv_non_lin': 'funMagnitude', 'pooling_function': 'average-pool', 'conv_const': 1.50, 'learn_const': True},}
    
    runNum = runNum % len(tests)
    hyperParam.poolingFun                              = tests[runNum]['pooling_function']
    hyperParam.non_linearity['FC']['type_of_nonlin']   = tests[runNum]['FC_non_lin']
    hyperParam.non_linearity['conv']['type_of_nonlin'] = tests[runNum]['conv_non_lin']
    hyperParam.non_linearity['conv']['const']          = tests[runNum]['conv_const']
    hyperParam.use_trainable_const                     = tests[runNum]['learn_const']
    
    
    # pooling_function = ['average-pool', 'max-pool']#, 'stride-pooling']
    # FC_nonlin = ['relu', 'identity']
    # conv_nonlin = ['relu', 'identity']#, 'funMagnitude']
    
    # options_iterating_over = 1
    # hyperParam.poolingFun = pooling_function[(runNum / options_iterating_over) % len(pooling_function)]
    # options_iterating_over = options_iterating_over * len(pooling_function)
    
    # hyperParam.non_linearity['FC']['type_of_nonlin'] = FC_nonlin[(runNum / options_iterating_over) % len(FC_nonlin)]
    # options_iterating_over = options_iterating_over * len(FC_nonlin)
    
    # hyperParam.non_linearity['conv']['type_of_nonlin'] = conv_nonlin[(runNum / options_iterating_over) % len(conv_nonlin)]
    
    createReadMe(hyperParam)
    return hyperParam
    
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