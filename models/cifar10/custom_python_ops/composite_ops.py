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
    
def applyConstantToMagnitudeFast(c, magFun = noEffectApplyConstant, magConstant = 1.0):
    epsilon = 1e-6
    
    mag = tf.abs(c)
    magAfterConstant = magFun(mag, magConstant) / (mag + epsilon)
    
    magCompl = tf.complex(magAfterConstant, tf.zeros(magAfterConstant.shape))
    
    return magCompl * c
    
def applyConstantToComplexPolar(c, magFun = noEffectApplyConstant, magConstant = 1.0, angleFun = noEffectApplyConstant, angleConstant = 1.0, reNormalizeAngle = False, anglePositiveValued = False):
    mag = tf.abs(c)
    pha = tf_angle(c)
    
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