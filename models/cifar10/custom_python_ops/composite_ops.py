import numpy as np
import tensorflow as tf
from custom_ops import tf_arctan2


def tf_angle(c):
    return tf_arctan2(tf.imag(c), tf.real(c))

def sqrtMagnitude(c):
    mag = tf.abs(c)
    pha = tf_angle(c)
    
    sqrtmag = tf.sqrt(tf.nn.relu(mag))

    magCompl = tf.complex(sqrtmag, tf.zeros(sqrtmag.shape))
    phaCompl = tf.complex(tf.zeros(pha.shape), pha)
    
    return magCompl * tf.exp(phaCompl)
    
def powMagnitude(c, power):
    # mag = tf.abs(c)
    # pha = tf_angle(c)
    
    # sqrtmag = tf.pow(tf.nn.relu(mag), power)

    # magCompl = tf.complex(sqrtmag, tf.zeros(sqrtmag.shape))
    # phaCompl = tf.complex(tf.zeros(pha.shape), pha)
    
    # return magCompl * tf.exp(phaCompl)
    return applyConstantToComplex(c, magFun = tf.pow, magConstant = power)
    
def noEffectApplyConstant(c, constant):
    return c
    
def applyConstantToComplex(c, magFun = noEffectApplyConstant, magConstant = 1.0, angleFun = noEffectApplyConstant, angleConstant = 1.0, reNormalizeAngle = False):
    mag = tf.abs(c)
    pha = tf_angle(c)
    
    magAfterConstant = magFun(tf.nn.relu(mag), magConstant)
    phaAfterConstant = angleFun(pha, angleConstant)
    if reNormalizeAngle:
        phaAfterConstant = tf.mod(phaAfterConstant + math.pi, 2*math.pi) - math.pi

    magCompl = tf.complex(magAfterConstant, tf.zeros(magAfterConstant.shape))
    phaCompl = tf.complex(tf.zeros(phaAfterConstant.shape), phaAfterConstant)
    
    return magCompl * tf.exp(phaCompl)