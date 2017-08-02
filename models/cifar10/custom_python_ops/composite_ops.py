import numpy as np
import tensorflow as tf
from custom_ops import tf_arctan2, tf_mod
import math

def tf_angle(c):
    return tf_arctan2(tf.imag(c), tf.real(c))

def powComplex(c, power):
    return applyConstantToComplex(c, magFun = tf.pow, magConstant = power, angleFun = tf.multiply, angleConstant = power, reNormalizeAngle = True)
    
def sqrtMagnitude(c):
    return powMagnitude(c, 0.5)
    
def powMagnitude(c, power):
    return applyConstantToComplex(c, magFun = tf.pow, magConstant = power)
    
def noEffectApplyConstant(c, constant):
    return c
    
def applyConstantToComplex(c, magFun = noEffectApplyConstant, magConstant = 1.0, angleFun = noEffectApplyConstant, angleConstant = 1.0, reNormalizeAngle = False):
    mag = tf.abs(c)
    pha = tf_angle(c)
    
    magAfterConstant = magFun(tf.nn.relu(mag), magConstant)
    phaAfterConstant = angleFun(pha, angleConstant)
    if reNormalizeAngle:
        phaAfterConstant = tf_mod(phaAfterConstant + math.pi, 2*math.pi) - math.pi
    
    magCompl = tf.complex(magAfterConstant, tf.zeros(magAfterConstant.shape))
    phaCompl = tf.complex(tf.zeros(phaAfterConstant.shape), phaAfterConstant)
    
    return magCompl * tf.exp(phaCompl)