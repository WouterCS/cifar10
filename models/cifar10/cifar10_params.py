import os.path

def main(runNum):
    
    directory = '/results/run-%d' % runNum
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    class hyperParameters:
        def __init__(self):
            self.datasetname = 'CIFAR-10'
            self.FCnonLin = 'identity' # 'relu'   'powMagnitude'
            self.FCnonLinMag = 0.9
            self.convNonLin = 'identity' # 'relu'   'powMagnitude'
            self.convNonLinMag = 0.9
            self.poolingFun = 'max-pool' # 'average-pool'
            self.max_steps = 1000000
            self.eval_frequency = 1000
    
    hyperParam = hyperParameters()
    hyperParam.max_steps = 300
    hyperParam.eval_frequency = 100

    if runNum == 1:
        hyperParam.convNonLin = 'powMagnitude'
        hyperParam.convNonLinMag = 0.9
    if runNum == 2:
        hyperParam.convNonLin = 'relu'
    if runNum == 3:
        hyperParam.convNonLin = 'identity'
    if runNum == 3:
        hyperParam.convNonLin = 'powMagnitude'
        hyperParam.convNonLinMag = 0.8
        
    return hyperParam
    
def createReadMe(hyperParam, directory):
    with open(directory + '/README.txt', 'wb') as f:
        print('Dataset: %s' % hyperParam.datasetname)
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