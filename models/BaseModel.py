import logging
from abc import abstractmethod
import tensorflow as tf

try:
    # Capirca uses Google's abseil-py library, which uses a Google-specific
    # wrapper for logging. That wrapper will write a warning to sys.stderr if
    # the Google command-line flags library has not been initialized.
    #
    # https://github.com/abseil/abseil-py/blob/pypi-v0.7.1/absl/logging/__init__.py#L819-L825
    #
    # This is not right behavior for Python code that is invoked outside of a
    # Google-authored main program. Use knowledge of abseil-py to disable that
    # warning; ignore and continue if something goes wrong.
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass


class BaseModel:
    def __init__(self,dataset, z0_size, z1_size, y_size, x_size, args):
        self.log = logging.getLogger(__name__)
        self.dataset = dataset
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.z0_size = z0_size
        self.z1_size = z1_size
        self.py = y_size
        self.px = x_size
        self.args = args
        self.batch_size = args.batch_size
        self.number_of_interations = args.iterations

    def encoder(self, tf_X, condvae=False):
        e_1 = tf.contrib.layers.fully_connected(tf_X, 256, activation_fn=tf.nn.softplus, biases_initializer=None,
                                                scope="e_1")
        e_2 = tf.contrib.layers.fully_connected(e_1, 256, activation_fn=tf.nn.softplus, biases_initializer=None,
                                                scope="e_2")

        mu = tf.contrib.layers.fully_connected(e_2, (self.z0_size+self.z1_size), activation_fn=None, biases_initializer=None,
                                               scope="mu")  ## latent means


        return mu

    def condVAEEncoder(self, tf_X, condvae=False):
        e_1 = tf.contrib.layers.fully_connected(tf_X, 256, activation_fn=tf.nn.softplus, biases_initializer=None,
                                                    scope="e_1")
        e_2 = tf.contrib.layers.fully_connected(e_1, 256, activation_fn=tf.nn.softplus, biases_initializer=None,
                                                    scope="e_2")

        mu = tf.contrib.layers.fully_connected(e_2, (self.z0_size), activation_fn=None,
                                                   biases_initializer=None,
                                                   scope="mu")  ## latent means


        z_log_sigma_sq = tf.contrib.layers.fully_connected(e_2, (self.z0_size),
                                                                   activation_fn=None, biases_initializer=None,
                                                                   scope="log_sig")  ## latent means

        return mu, z_log_sigma_sq



    def decoder(self,z):
        d_1 = tf.contrib.layers.fully_connected(z, 256, activation_fn=tf.nn.softplus, scope="d_1")
        d_2 = tf.contrib.layers.fully_connected(d_1, 256, activation_fn=tf.nn.softplus, scope="d_2")
        x_mu = tf.contrib.layers.fully_connected(d_2, self.px, activation_fn=None, scope="x_mu")
        return x_mu

    def predictor(self,z):
        d_1_i = tf.contrib.layers.fully_connected(z, 256, activation_fn=tf.nn.softplus, scope="z_1_y_d1")
        d_2_i = tf.contrib.layers.fully_connected(d_1_i, 256, activation_fn=tf.nn.softplus, scope="z_1_y_d2")
        y_mu = tf.contrib.layers.fully_connected(d_2_i, self.py, activation_fn=None, scope="y_mu")
        return y_mu


    @abstractmethod
    def buildModel(self):
        pass

    @abstractmethod
    def optimizeModel(self):
        pass

    @abstractmethod
    def evaluateModel(self):
        pass