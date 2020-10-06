import logging

import numpy as np
import tensorflow as tf
from npeet import entropy_estimators as ee

from models.BaseModel import BaseModel
import Plotter
import Strings

class VAEModel(BaseModel):
    def __init__(self,dataset, z0_size, z1_size, y_size, x_size, args):
        super().__init__(dataset, z0_size, z1_size, y_size, x_size, args)

        self.log = logging.getLogger(__name__)
        self.lm = 0.7

    def buildModel(self):
        tf.reset_default_graph()
        self.sigma_t = 1
        self.log_sigma = tf.Variable((0.0, 0.0, 0.0), (self.z1_size + self.z0_size), name="log_sigma")
        self.log_sigma_t = tf.constant(0.0)

        self.tf_X = tf.placeholder(tf.float32, [None, self.py])
        self.tf_Y = tf.placeholder(tf.float32, [None, self.py])
        self.lagMul = tf.placeholder(tf.float32, [1, 1], name='lagMul')
        pz = self.z1_size+self.z0_size


        self.mu = self.encoder(self.tf_X)

        n = tf.shape(self.mu)[0]

        eps = tf.random_normal(tf.stack((n, pz)), 0, tf.exp(self.log_sigma), dtype=tf.float32)  ## Adding a random number
        self.z = tf.add(eps, self.mu)  ## The sampled z

        self.x_mu = self.decoder(self.z)
        self.y_mu = self.predictor(self.z)

        self.x_reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_mu - self.tf_X), 1))
        self.x_mae = tf.reduce_mean(tf.reduce_sum(tf.abs(self.x_mu - self.tf_X), 1))

        self.y_reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_mu - self.tf_Y), 1))
        self.y_mae = tf.reduce_mean(tf.reduce_sum(tf.abs(self.y_mu - self.tf_Y), 1))

        KL = self.log_sigma_t - self.log_sigma + (tf.square(self.mu) + tf.square(tf.exp(self.log_sigma))) / (
        2.0 * tf.square(tf.exp(self.log_sigma_t))) - 0.5

        self.latent_loss = tf.reduce_mean(tf.reduce_sum(KL, 1))

        self.loss = self.latent_loss + (self.x_reconstr_loss + 3 * self.y_reconstr_loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
            self.loss)  ## minimize reconstruction losses and minimize I(z.0,y)

        self.saver = tf.train.Saver()



    def optimizeModel(self):

        x_l_t = np.inf
        y_l_t = np.inf

        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for iter in range(self.number_of_interations):

                x_batch, y_batch, t_batch = self.dataset.next_batch(self.batch_size)

                _, mu_out, ll_out, rl_out, log_sigma_out, yl_out = sess.run(
                    [self.optimizer, self.mu, self.latent_loss, self.x_reconstr_loss, self.log_sigma, self.y_reconstr_loss],
                    feed_dict={self.tf_X: x_batch, self.tf_Y: y_batch, self.lagMul: np.asarray([[self.lm]])})

                if ((iter) % 300 == 0) and iter > 1:

                    mu_out = sess.run(
                        self.mu,
                        feed_dict={self.tf_X: x_batch, self.tf_Y: y_batch, self.lagMul: np.asarray([[self.lm]])})

                    x_mae_out, y_o = sess.run([self.x_mae, self.y_mae],
                                              feed_dict={self.tf_X: x_batch, self.tf_Y: y_batch, self.lagMul: np.asarray([[self.lm]]),
                                                         self.z: mu_out})

                    print(iter, x_mae_out, y_o)
                    if y_o <= y_l_t and x_mae_out <= x_l_t:
                        y_l_t = y_o
                        x_l_t = x_mae_out
                        print("save model")
                        save_path = self.saver.save(sess, self.args.save_path)

                        self.lm = self.lm * 1.03



    def evaluateModel(self):
        with tf.Session() as sess:
            # Restore variables from disk.
            self.saver.restore(sess, self.args.pretrained)

            x,y,_,bins = self.dataset.get_eval_data()

            mu_out, x_rec, y_rec = sess.run(
                [self.mu, self.x_mu, self.y_mu],
                feed_dict={self.tf_X: x, self.tf_Y: y, self.lagMul: np.asarray([[self.lm]])})

            x_mae_out, y_o, z0 = sess.run([self.x_mae, self.y_mae, self.z],
                                          feed_dict={self.tf_X: x, self.tf_Y: y, self.lagMul: np.asarray([[self.lm]]), self.z: mu_out})

            mi = ee.mi(z0, y, k=20)
            self.log.info("Mutual Information: %f, X MAE: %f, Y MAE: %f" % (mi, x_mae_out, y_o))


            Plotter.plot2DData(y_rec[:, :], bins[:], self.args.plot_path, Strings.VAE_Y)
            Plotter.plot2DLatentSpace(z0[:, 0], z0[:, 1], bins[:], self.args.plot_path, Strings.VAE_LATENT)
            Plotter.plot2DData(x_rec[:, :], bins[:], self.args.plot_path, Strings.VAE_X)
