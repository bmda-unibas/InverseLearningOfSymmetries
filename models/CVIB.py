import logging

import numpy as np
import tensorflow as tf
from npeet import entropy_estimators as ee

from models.BaseModel import BaseModel
import Plotter
import Strings

np.random.seed(1234)

class CVIBModel(BaseModel):
    def __init__(self,dataset, z0_size, z1_size, y_size, x_size, args):
        super().__init__(dataset, z0_size, z1_size, y_size, x_size, args)

        self.log = logging.getLogger(__name__)
        self.lm = 0.7

    # MI regulariser taken from Moyer et al.

    def all_pairs_gaussian_kl(self, mu, sigma, add_third_term=False):

        sigma_sq = tf.square(sigma) + 1e-8

        # mu is [batchsize x dim_z]
        # sigma is [batchsize x dim_z]

        sigma_sq_inv = tf.math.reciprocal(sigma_sq)
        # sigma_inv is [batchsize x sizeof(latent_space)]

        #
        # first term
        #

        # dot product of all sigma_inv vectors with sigma
        # is the same as a matrix mult of diag
        first_term = tf.matmul(sigma_sq, tf.transpose(sigma_sq_inv))

        #
        # second term
        #

        # TODO: check this
        # REMEMBER THAT THIS IS SIGMA_1, not SIGMA_0

        r = tf.matmul(mu * mu, tf.transpose(sigma_sq_inv))
        # r is now [batchsize x batchsize] = sum(mu[:,i]**2 / Sigma[j])

        r2 = mu * mu * sigma_sq_inv
        r2 = tf.reduce_sum(r2, 1)
        # r2 is now [batchsize, 1] = mu[j]**2 / Sigma[j]

        # squared distance
        # (mu[i] - mu[j])\sigma_inv(mu[i] - mu[j]) = r[i] - 2*mu[i]*mu[j] + r[j]
        # uses broadcasting
        second_term = 2 * tf.matmul(mu, tf.transpose(mu * sigma_sq_inv))
        second_term = r - second_term + tf.transpose(r2)

        ##uncomment to check using tf_tester
        # return second_term

        #
        # third term
        #

        # log det A = tr log A
        # log \frac{ det \Sigma_1 }{ det \Sigma_0 } =
        #   \tr\log \Sigma_1 - \tr\log \Sigma_0
        # for each sample, we have B comparisons to B other samples...
        #   so this cancels out

        if (add_third_term):
            r = tf.reduce_sum(tf.math.log(sigma_sq), 1)
            r = tf.reshape(r, [-1, 1])
            third_term = r - tf.transpose(r)
        else:
            third_term = 0

        # - tf.reduce_sum(tf.log(1e-8 + tf.square(sigma)))\
        # the dim_z ** 3 term comes from
        #   -the k in the original expression
        #   -this happening k times in for each sample
        #   -this happening for k samples
        # return 0.5 * ( first_term + second_term + third_term - dim_z )
        return 0.5 * (first_term + second_term + third_term)

    #
    # kl_conditional_and_marg
    #   \sum_{x'} KL[ q(z|x) \| q(z|x') ] + (B-1) H[q(z|x)]
    #

    # def kl_conditional_and_marg(args):
    def kl_conditional_and_marg(self, z_mean, z_log_sigma_sq, dim_z):
        z_sigma = tf.exp(0.5 * z_log_sigma_sq)
        all_pairs_GKL = self.all_pairs_gaussian_kl(z_mean, z_sigma, True) - 0.5 * dim_z
        return tf.reduce_mean(all_pairs_GKL)

    def buildModel(self):
        tf.reset_default_graph()

        self.tf_X = tf.placeholder(tf.float32, [None, self.py])
        self.tf_Y = tf.placeholder(tf.float32, [None, self.py])
        self.lagMul = tf.placeholder(tf.float32, [1, 1], name='lagMul')
        pz = self.z0_size


        self.mu, self.z_log_sigma_sq = self.condVAEEncoder(self.tf_X)

        n = tf.shape(self.mu)[0]

        eps = tf.random_normal((n, pz), 0, 1, dtype=tf.float32)  # Adding a random number
        self.z_plain = tf.add(self.mu, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))  # The sampled z

        self.z = tf.concat([self.z_plain, self.tf_Y], 1)
        self.x_mu = self.decoder(self.z)

        self.x_reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_mu - self.tf_X), 1))
        self.x_mae = tf.reduce_mean(tf.reduce_sum(tf.abs(self.x_mu - self.tf_X), 1))

        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.mu) - tf.exp(self.z_log_sigma_sq), 1)
        self.latent_loss = tf.reduce_mean(self.latent_loss)


        self.loss = 0.1*self.latent_loss + (1+1) * (self.x_reconstr_loss) + 1*self.kl_conditional_and_marg(self.mu, self.z_log_sigma_sq, pz)

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

                _, mu_out, ll_out, rl_out, log_sigma_out = sess.run(
                    [self.optimizer, self.mu, self.latent_loss, self.x_reconstr_loss, self.log_sigma],
                    feed_dict={self.tf_X: x_batch, self.tf_Y: y_batch, self.lagMul: np.asarray([[self.lm]])})

                if ((iter) % 300 == 0) and iter > 1:

                    mu_out = sess.run(
                        self.mu,
                        feed_dict={self.tf_X: x_batch, self.tf_Y: y_batch, self.lagMul: np.asarray([[self.lm]])})

                    x_mae_out = sess.run([self.x_mae],
                                              feed_dict={self.tf_X: x_batch, self.tf_Y: y_batch, self.lagMul: np.asarray([[self.lm]]),
                                                         self.z: mu_out})

                    print(iter, x_mae_out)
                    if x_mae_out <= x_l_t:
                        x_l_t = x_mae_out
                        print("save model")
                        save_path = self.saver.save(sess, self.args.save_path)

                        self.lm = self.lm * 1.03



    def evaluateModel(self):
        with tf.Session() as sess:
            # Restore variables from disk.
            self.saver.restore(sess, self.args.pretrained)
            x,y,_,bins = self.dataset.get_eval_data()

            mu_out, x_rec = sess.run(
                [self.mu, self.x_mu],
                feed_dict={self.tf_X: x, self.tf_Y: y, self.lagMul: np.asarray([[self.lm]])})

            x_mae_out, z0 = sess.run([self.x_mae, self.z_plain],
                                          feed_dict={self.tf_X: x, self.tf_Y: y, self.z_plain: mu_out})

            mi = ee.mi(z0, y, k=20)
            self.log.info("Mutual Information: %f, X MAE: %f" % (mi, x_mae_out))

            # tmp_idx = np.random.choice(10000, 1000)
            #
            # Plotter.plot2DData(x_rec[tmp_idx, :], bins[tmp_idx], self.args.plot_path, "moyer-x-rec_0.pdf")



