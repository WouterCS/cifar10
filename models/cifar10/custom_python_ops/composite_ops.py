import numpy as np
import tensorflow as tf
from custom_ops import tf_arctan2


def tf_angle(c):
    return tf.atan2(tf.imag(c), tf.real(c))

def sqrtMagnitude(c):
    return powMagnitude(c, 0.5)
    
def powMagnitude(c, power):
    return applyConstantToComplex(c, magFun = tf.pow, magConstant = power)
    
def noEffectApplyConstant(c, constant):
    return c
    
def applyConstantToComplex(c, magFun = noEffectApplyConstant, magConstant = 1.0, angleFun = noEffectApplyConstant, angleConstant = 1.0, reNormalizeAngle = False):
    mag = tf.abs(c)
    pha = tf_angle(c)
    
    #magAfterConstant = tf.nn.relu(magFun(mag, magConstant))
    #phaAfterConstant = angleFun(pha, angleConstant)
    #if reNormalizeAngle:
    #    phaAfterConstant = tf.mod(phaAfterConstant + math.pi, 2*math.pi) - math.pi
    magCompl = tf.add(mag, 0)
    phaCompl = pha
    
    magCompl = tf.complex(magAfterConstant, tf.zeros(magAfterConstant.shape))
    phaCompl = tf.complex(tf.zeros(phaAfterConstant.shape), phaAfterConstant)
    
    return magCompl * tf.exp(phaCompl)