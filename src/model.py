from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from module import *
from utils import *

class U_net(object):
    def __init__(self,sess,args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.dataset_dir = args.dataset_dir
        self.net = net
        OPTIONS = namedtuple('OPTIONS', 'batch_size gf_dim image_size output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.ngf,args.fine_size,
                                      args.output_nc,
                                      args.phase == 'train'))
        
        self._build_model()
        self.saver = tf.compat.v1.train.Saver()
        self.pool = ImagePool(args.max_size)
        self.plot = []

    def _build_model(self):
        self.data = tf.compat.v1.placeholder(tf.float32,[None,self.image_size,self.image_size,self.input_c_dim + self.output_c_dim],name='Input_data')
        self.input_data = self.data[:,:,:,:self.input_c_dim]
        self.real_data = self.data[:,:,:,self.input_c_dim:self.input_c_dim + self.output_c_dim]
        self.output_data = self.net(self.input_data,self.options,False,name="generator")
        self.loss = mae_criterion(self.output_data,self.real_data)

        self.input_test = tf.compat.v1.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,self.input_c_dim], name='input_test')
        self.output_test = self.net(self.input_test,self.options,False,name='generate')

        t_vars = tf.compat.v1.trainable_variables()

        self.vars = [var for var in t_vars]
        for var in t_vars: print(var.name)

    def train(self,args):
        self.lr = tf.compat.v1.placeholder(tf.float32, None, name='learning_rate')
        self.optim = tf.compat.v1.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.loss, var_list=self.vars)

        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)
        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        
        for epoch in range(args.epoch):
            flies = os.listdir(self.dataset_dir)
            for file in flies:
                input_data = glob('./datasets/{}/data/*.*'.format(file))
                real_data = glob('./datasets/{}/label/*.*'.format(file))

                batch_idxs = min(min(len(input_data), len(real_data)), args.train_size) // self.batch_size
                lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            
                for idx in range(0, batch_idxs):
                    batch_files = list(zip(input_data[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       real_data[idx * self.batch_size:(idx + 1) * self.batch_size]))
                    batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
                    batch_images = np.array(batch_images).astype(np.float32)
                    output_data, _ ,loss = self.sess.run([self.output_data,self.optim,self.loss],feed_dict={self.data: batch_images, self.lr: lr})
                    self.plot.append(loss)

                    counter += 1
                    print(("Epoch: [%2d] Patient:[%s] [%4d/%4d] time: %4.4f" % (
                        epoch,file,idx, batch_idxs, time.time() - start_time)))
                    try:
                        print("loss is %f"%loss)
                    except:
                        print("The loss is too large")
                    
                    if np.mod(counter, args.save_freq) == 2:
                        self.save(args.checkpoint_dir, counter)
        plt.plot(self.plot)
        plt.show()
    def save(self,checkpoint_dir,step):
        model_name = "U_net.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
    
    def test(self,args):
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)
        sample_files = glob('./datasets/test/*.*')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            
            image_path = os.path.join(args.test_dir,
                                      '{}'.format(os.path.basename(sample_file)))
            fake_img = self.sess.run(self.output_test, feed_dict={self.input_test: sample_image})
            save_images(fake_img, [1, 1], image_path)
    # def test(self,args):
    #     init_op = tf.compat.v1.global_variables_initializer()
    #     self.sess.run(init_op)
    #     sample_files = glob('./datasets/test/data/*.*')
    #     real_files = glob('./datasets/test/label/*.*')

    #     if self.load(args.checkpoint_dir):
    #         print(" [*] Load SUCCESS")
    #     else:
    #         print(" [!] Load failed...")
    #     batch_idxs = min(min(len(sample_files), len(real_files)), args.train_size) // self.batch_size

            
    #     for idx in range(0, batch_idxs):
    #         batch_files = list(zip(sample_files[idx * self.batch_size:(idx + 1) * self.batch_size],
    #                                    real_files[idx * self.batch_size:(idx + 1) * self.batch_size]))
    #         batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
    #         batch_images = np.array(batch_images).astype(np.float32)
    #         output_data,loss = self.sess.run([self.output_data,self.loss],feed_dict={self.data: batch_images})
    #         print(loss*7560)









