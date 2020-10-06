import logging

import numpy as np
import tensorflow as tf
from npeet import entropy_estimators as ee

from models.BaseModel import BaseModel
import Plotter
import Strings

class STIBModel(BaseModel):
    def __init__(self,dataset, z0_size, z1_size, y_size, x_size, args):
        super().__init__(dataset, z0_size, z1_size, y_size, x_size, args)

        self.log = logging.getLogger(__name__)
        self.lm = 0.7


    def ir_loss_calc(self, y_mur_sg, target):

        def normalize_with_moments(x, axes=0, epsilon=1e-8):
            mv = tf.nn.moments(x, axes=axes)
            mean = mv[0]
            variance = mv[1]
            x_normed = (x - mean) / tf.sqrt(variance + epsilon)  # epsilon to avoid dividing by zero
            return x_normed

        def tf_cov(x):
            x = normalize_with_moments(x)
            fact = tf.cast(tf.shape(x)[0] - 1, tf.float32)
            cov = tf.matmul(tf.transpose(x), x) / fact
            return cov

        cmat = tf_cov(tf.concat([y_mur_sg, target], -1))
        cmat_1 = tf_cov(tf.concat([y_mur_sg], -1))
        cmat_2 = tf_cov(tf.concat([target], -1))
        ir_loss = tf.log(tf.linalg.det(cmat_1) * tf.linalg.det(cmat_2) / tf.linalg.det(cmat))
        return ir_loss


    def buildBijectionNetwork(self, tf_Y):
        d_y_1 = tf.contrib.layers.fully_connected(tf_Y, 256, activation_fn=tf.nn.softplus, scope="y_yhat_d1")
        d_y_2 = tf.contrib.layers.fully_connected(d_y_1, 256, activation_fn=tf.nn.softplus, scope="y_yhat_d2")
        y_hat = tf.contrib.layers.fully_connected(d_y_2, self.py, activation_fn=None, scope="yhat")

        d_r_1 = tf.contrib.layers.fully_connected(y_hat, 256, activation_fn=tf.nn.softplus, scope="yhat_y_d1")
        d_r_2 = tf.contrib.layers.fully_connected(d_r_1, 256, activation_fn=tf.nn.softplus, scope="yhat_y_d2")
        target_hat = tf.contrib.layers.fully_connected(d_r_2, self.py, activation_fn=None, scope="target_hat")
        return target_hat, y_hat


    def buildIvarianceNetwork(self,z_0):
        d_1_ir_sg = tf.contrib.layers.fully_connected(z_0, 256, activation_fn=tf.nn.softplus, scope="z_0_y_d1")
        d_2_ir_sg = tf.contrib.layers.fully_connected(d_1_ir_sg, 256, activation_fn=tf.nn.softplus, scope="z_0_y_d2")
        y_mur_sg = tf.contrib.layers.fully_connected(d_2_ir_sg, self.py, activation_fn=None, scope="y_mu_irr")
        return y_mur_sg

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

        self.z_1 = self.z[:, 2]
        self.z_1 = tf.reshape(self.z_1, shape=(-1, 1))
        self.z_0 = self.z[:, 0:2]

        self.x_mu = self.decoder(self.z)
        self.y_mu = self.predictor(self.z_1)

        y_mur_sg = self.buildIvarianceNetwork(self.z_0)

        target_hat, y_hat = self.buildBijectionNetwork(self.tf_Y)

        self.bi_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.tf_Y - target_hat), 1))
        self.ir_loss = self.ir_loss_calc(y_mur_sg, y_hat)



        self.x_reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_mu - self.tf_X), 1))
        self.x_mae = tf.reduce_mean(tf.reduce_sum(tf.abs(self.x_mu - self.tf_X), 1))

        self.y_reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_mu - self.tf_Y), 1))
        self.y_mae = tf.reduce_mean(tf.reduce_sum(tf.abs(self.y_mu - self.tf_Y), 1))

        KL = self.log_sigma_t - self.log_sigma + (tf.square(self.mu) + tf.square(tf.exp(self.log_sigma))) / (
        2.0 * tf.square(tf.exp(self.log_sigma_t))) - 0.5

        self.latent_loss = tf.reduce_mean(tf.reduce_sum(KL, 1))

        self.loss = self.latent_loss + self.lagMul * (self.x_reconstr_loss + 1 * self.y_reconstr_loss + 4e2 * self.ir_loss)

        tvars = tf.trainable_variables()

        ## alternating optimization
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss, var_list=tvars[
                                                                                       0:18])  ## minimize reconstruction losses and minimize I(z.0,y)
        self.optimizer_ir = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(-3e3 * (self.ir_loss - 0.01 * self.bi_loss),
                                                                           var_list=tvars[
                                                                                    18:34])  ## maximize I(z.0,y) and minimize bijection loss #- bi_loss

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

                _ = sess.run(self.optimizer_ir,
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

            x_mae_out, y_o, z1, z0 = sess.run([self.x_mae, self.y_mae, self.z_1, self.z_0],
                                          feed_dict={self.tf_X: x, self.tf_Y: y, self.lagMul: np.asarray([[self.lm]]), self.z: mu_out})

            mi = ee.mi(z0, y, k=20)
            self.log.info("Mutual Information: %f, X MAE: %f, Y MAE: %f" % (mi, x_mae_out, y_o))


            Plotter.plot2DData(y_rec[:, :], bins[:], self.args.plot_path, Strings.STIB_Y)
            Plotter.plot2DLatentSpace(z0[:, 0], z1[:], bins[:], self.args.plot_path, Strings.STIB_LATENT)
            Plotter.plot2DData(x_rec[:, :], bins[:], self.args.plot_path, Strings.STIB_X)
