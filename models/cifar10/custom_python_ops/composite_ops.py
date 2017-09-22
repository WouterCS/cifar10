import numpy as np
import tensorflow as tf
from custom_ops import tf_arctan2, tf_mod
import math

def tf_angle(c):
    return tf_arctan2(tf.imag(c), tf.real(c))

def powComplex(c, power):
    return applyConstantToComplexPolar(c, magFun = tf.pow, magConstant = power, angleFun = tf.multiply, angleConstant = power, reNormalizeAngle = True)
    
def sqrtMagnitude(c):
    return powMagnitude(c, 0.5)
    
def powMagnitude(c, power):
    return applyConstantToComplexPolar(c, magFun = tf.pow, magConstant = power)
    
def noEffectApplyConstant(c, constant):
    return c
    
def applyTaylerToMagnitude(c, coeffs):
    
    def taylorFunction(x, taylor_coeffs):
        x_out = taylor_coeffs[0]
        for coef in taylor_coeffs[1:]:
            x_out = x_out * x + coef
            
        #with tf.control_dependencies([tf.assert_non_negative(tf.reduce_max(taylor_coeffs))]):
        x_out = tf.nn.relu(x_out)
        return x_out
    
    return applyConstantToMagnitudeFast(c, taylorFunction, coeffs)
    
def applyConstantToMagnitudeFast(c, magFun = noEffectApplyConstant, magConstant = 1.0):
    epsilon = 1e-7
    
    mag = tf.abs(c)
    
    # get histogram of input:
    def getHist(v, k):
        v = tf.reshape(v, [-1])
        s = v.shape[0]
        v = tf.nn.top_k(v, s)
        out_v = map(lambda i: v.values[(s*i)/k], range(k))
        out_v.append(tf.reduce_max(v))
        out_v.append(tf.reduce_min(mag))
        return out_v
            
    mag = tf.Print(mag, getHist(mag, 10), 'Histogram: ')
    
    magAfterConstant = magFun(tf.nn.relu(mag), magConstant) / (mag + epsilon)
    
    with tf.control_dependencies([tf.assert_positive(tf.reduce_max(magAfterConstant))]):
        magCompl = tf.complex(magAfterConstant, tf.zeros(magAfterConstant.shape))
    
    return magCompl * c
    
def applyConstantToComplexPolar(c, magFun = noEffectApplyConstant, magConstant = 1.0, angleFun = noEffectApplyConstant, angleConstant = 1.0, reNormalizeAngle = False, anglePositiveValued = False):
    mag = tf.abs(c)
    pha = tf_angle(c)
    
    #mag = tf.Print(mag, [tf.reduce_max(mag), tf.reduce_min(mag), tf.reduce_mean(mag) ], message = 'Magnitudes before power applied: ')
    
    magAfterConstant = magFun(tf.nn.relu(mag), magConstant)
    if anglePositiveValued:
        pha = tf.nn.relu(pha + math.pi)
        phaAfterConstant = angleFun(pha, angleConstant)
        phaAfterConstant = phaAfterConstant - math.pi
    else:
        phaAfterConstant = angleFun(pha, angleConstant)
    if reNormalizeAngle:
        phaAfterConstant = tf_mod(phaAfterConstant + math.pi, 2*math.pi) - math.pi
    
    magCompl = tf.complex(magAfterConstant, tf.zeros(magAfterConstant.shape))
    phaCompl = tf.complex(tf.zeros(phaAfterConstant.shape), phaAfterConstant)
    
    return magCompl * tf.exp(phaCompl)
    
    
def applyConstantToComplexCart(c, realFun = noEffectApplyConstant, realConstant = 1.0, imagFun = noEffectApplyConstant, imagConstant = 1.0):
    return tf.complex(realFun(tf.real(c), realConstant), imagFun(tf.imag(c), imagConstant))