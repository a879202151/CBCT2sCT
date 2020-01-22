from __future__ import division
import tensorflow as tf 
from ops import *
from utils import *

def net(image, options, reuse=False, name="net"):
    with tf.compat.v1.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred
# def net(image, options, reuse=False, name="net"):
#     with tf.compat.v1.variable_scope(name):
#         # image is 410 x 410 x input_c_dim
#         if reuse:
#             tf.compat.v1.get_variable_scope().reuse_variables()
#         else:
#             assert tf.compat.v1.get_variable_scope().reuse is False
#         rcu_1 = RCU_Block(image,options.gf_dim,name='RCU_1',is_training=options.is_training)
#         cp_1 = CP_Block(rcu_1,options.gf_dim*2,name='CP_1')
#         rcu_2 = RCU_Block(cp_1,options.gf_dim*2,name='RCU_2',is_training=options.is_training)
#         cp_2 = CP_Block(rcu_2,options.gf_dim*4,name='CP_2')
#         rcu_3 = RCU_Block(cp_2,options.gf_dim*4,name='RCU_3',is_training=options.is_training)
#         cp_3 = CP_Block(rcu_3,options.gf_dim*8,name='CP_3')
#         rcu_4 = RCU_Block(cp_3,options.gf_dim*8,name='RCU_4',is_training=options.is_training)
#         cp_4 = CP_Block(rcu_4,options.gf_dim*16,name='CP_4')
#         rcu_5 = RCU_Block(cp_4,options.gf_dim*16,name='RCU_5',is_training=options.is_training)
#         cp_5 = CP_Block(rcu_5,options.gf_dim*32,name='CP_5')
#         rcu_6 = RCU_Block(cp_5,options.gf_dim*32,name='RCU_6',is_training=options.is_training)
#         du_1 = tf.concat([deconv2d(rcu_6,options.gf_dim*16,name='DU_1'),rcu_5],3)
#         rcu_7 = RCU_Block(du_1,options.gf_dim*16,name='RCU_7',is_training=options.is_training)
#         du_2 = tf.concat([deconv2d(rcu_7,options.gf_dim*8,name='DU_2'),rcu_4],3)
#         rcu_8 = RCU_Block(du_2,options.gf_dim*8,name='RCU_8',is_training=options.is_training)
#         du_3 = tf.concat([deconv2d(rcu_8,options.gf_dim*4,name='DU_3'),rcu_3],3)
#         rcu_9 = RCU_Block(du_3,options.gf_dim*4,name='RCU_9',is_training=options.is_training)
#         du_4 = tf.concat([deconv2d(rcu_9,options.gf_dim*2,name='DU_4'),rcu_2],3)
#         rcu_10 = RCU_Block(du_4,options.gf_dim*2,name='RCU_10',is_training=options.is_training)
#         du_5 = tf.concat([deconv2d(rcu_10,options.gf_dim,name='DU_5'),rcu_1],3)
#         rcu_11 = RCU_Block(du_5,options.gf_dim,name='RCU_11',is_training=options.is_training)
        
#         return tf.nn.tanh(conv2d(rcu_11,options.output_c_dim,ks=1,s=1,name='conv'))


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

