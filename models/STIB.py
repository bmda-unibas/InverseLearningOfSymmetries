import logging

import numpy as np
import tensorflow as tf
from npeet import entropy_estimators as ee

from models.BaseModel import BaseModel
import Plotter
import Strings

np.random.seed(1234)

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

        px = 2

        pz = 3
        py = 2

        sigma_t = 1

        self.log_sigma = tf.Variable((0.0, 0.0, 0.0), pz, name="log_sigma")
        log_sigma_t = tf.constant(0.0)

        self.tf_X = tf.placeholder(tf.float32, [None, px])
        self.tf_Y = tf.placeholder(tf.float32, [None, py])

        self.lagMul = tf.placeholder(tf.float32, [1, 1], name='lagMul')

        self.mu = self.encoder(self.tf_X)

        n = tf.shape(self.mu)[0]

        eps = tf.random_normal(tf.stack((n, pz)), 0, tf.exp(self.log_sigma), dtype=tf.float32)  ## Adding a random number

        self.z = tf.add(eps, self.mu)

        self.x_mu = self.decoder(self.z)

        self.x_mae = tf.reduce_mean(tf.reduce_sum(tf.abs(self.x_mu - self.tf_X), 1))

        self.x_reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_mu - self.tf_X), 1))

        ## partition z
        z_1 = self.z[:, 2]
        self.z_1 = tf.reshape(z_1, shape=(-1, 1))
        self.z_0 = self.z[:, 0:2]

        self.y_mu = self.predictor(self.z_1)

        self.y_mur_sg = self.buildIvarianceNetwork(self.z_0)

        self.target_hat, self.y_hat = self.buildBijectionNetwork(self.tf_Y)

        self.y_reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_mu - self.tf_Y), 1))
        self.y_mae = tf.reduce_mean(tf.reduce_sum(tf.abs(self.y_mu - self.tf_Y), 1))

        self.y_irr_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_mur_sg - self.tf_Y), 1))
        self.y_irr_mae = tf.reduce_mean(tf.reduce_sum(tf.abs(self.y_mur_sg - self.tf_Y), 1))


        ## Mutual information based on Gaussian model for p(y.mur.sg, y.hat)
        #target = y_hat
        self.ir_loss = self.ir_loss_calc(self.y_mur_sg, self.y_hat)

        self.bi_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.tf_Y - self.target_hat), 1))

        KL = log_sigma_t - self.log_sigma + (tf.square(self.mu) + tf.square(tf.exp(self.log_sigma))) / (
        2.0 * tf.square(tf.exp(log_sigma_t))) - 0.5

        self.latent_loss = tf.reduce_mean(tf.reduce_sum(KL, 1))

        self.loss = self.latent_loss + self.lagMul * (self.x_reconstr_loss + 1 * self.y_reconstr_loss + 4e2 * self.ir_loss)

        tvars = tf.trainable_variables()

        ## alternating optimization
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss, var_list=tvars[
                                                                                       0:18])  ## minimize reconstruction losses and minimize I(z.0,y)
        self.optimizer_ir = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(-3e3 * (self.ir_loss - 0.01 * self.bi_loss),
                                                                           var_list=tvars[
                                                                                    18:34])  ## maximize I(z.0,y) and minimize bijection loss #- bi_loss




    def optimizeModel(self):

        lm = 0.7

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        saver = tf.train.Saver(max_to_keep=1000000)

        x_l_t = np.inf
        y_l_t = np.inf

        x, y, _, bins = self.dataset.get_eval_data()

        with tf.Session(config=config) as sess:
            sess.run(tf.initialize_all_variables())

            for iter in range(150000):

                x_batch, y_batch, t_batch = self.dataset.next_batch(self.batch_size)

                _, mu_out, ll_out, rl_out, log_sigma_out, yl_out, irl_out, z0_out_tr = sess.run(
                    [self.optimizer, self.mu, self.latent_loss, self.x_reconstr_loss, self.log_sigma,
                     self.y_reconstr_loss, self.ir_loss, self.z_0],
                    feed_dict={self.tf_X: x_batch, self.tf_Y: y_batch, self.lagMul: np.asarray([[lm]])})

                _, irl_out_adv, z_out, y_mur_sg_out, bi_loss_out = sess.run(
                    [self.optimizer_ir, self.ir_loss, self.z, self.y_mur_sg, self.bi_loss],
                    feed_dict={self.tf_X: x_batch, self.tf_Y: y_batch, self.lagMul: np.asarray([[lm]])})

                if ((iter) % 300 == 0) and iter > 1:
                    self.log.info(
                        "Iteration: %d, Lambda: %.2f, I(x,t): %.2f, I(Z,X): %.2f, I(Z1,Y): %.2f" % (
                            iter, lm, ll_out, rl_out, yl_out))

                if iter % 300 == 0 and iter > 0:

                    mu_out_tmp = sess.run([self.mu],
                                          feed_dict={self.tf_X: x, self.tf_Y: y, self.lagMul: np.asarray([[lm]])})

                    x_rec, y_rec, y_irr_rec, ll_test, y_rec_test_loss, y_irr_test_loss, mi_loss, y_o, y_ir_o, y_hat_test, z0_out, x_test_loss, x_mae_out, bi_out = sess.run(
                        [self.x_mu, self.y_mu, self.y_mur_sg, self.latent_loss, self.y_reconstr_loss,
                         self.y_irr_rec_loss, self.ir_loss,
                         self.y_mae, self.y_irr_mae, self.y_hat, self.z_0, self.x_reconstr_loss, self.x_mae,
                         self.bi_loss],
                        feed_dict={self.tf_X: x, self.tf_Y: y, self.lagMul: np.asarray([[lm]]),
                                   self.z: np.reshape(mu_out_tmp, (10000, 3))})
                    self.log.info("I(Z0,Y): %.2f, BI Loss: %.2f, X MAE: %.2f, Y MAE: %.2f" % (ee.mi(z0_out, y, k=20), bi_out, x_mae_out, y_o))

                    if y_o <= y_l_t and x_mae_out <= x_l_t:
                        y_l_t = y_o
                        x_l_t = x_mae_out
                        self.log.info("saved model ...")
                        save_path = saver.save(sess, self.args.save_path)

                    lm = lm * 1.03



    def evaluateModel(self):
        with tf.Session() as sess:
            # Restore variables from disk.
            saver = tf.train.Saver()
            saver.restore(sess, self.args.pretrained)

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

            # tmp_idx = np.random.choice(10000, 1000)
            #
            # Plotter.plot2DData(x_rec[tmp_idx, :], bins[tmp_idx], self.args.plot_path, "plot-x-rec_4.pdf")
            # Plotter.plot2DData(y_rec[tmp_idx, :], bins[tmp_idx], self.args.plot_path, "plot-y-rec_4.pdf")
