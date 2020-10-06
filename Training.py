import logging

import Strings
from models.VAE import VAEModel
from models.STIBwoIR import STIBwoIRModel
from models.STIB import STIBModel
from models.CVAE import CVAEModel
from models.CVIB import CVIBModel


class Training:
    def __init__(self, args):
        self.log = logging.getLogger(__name__)
        self.args = args

    def train(self):
        self.log.info("Evaluating %s" % self.args.model)

        if self.args.model == Strings.VAE:

            vae = VAEModel(dataset=self.dataset, z0_size=2, z1_size=1, y_size=2, x_size=2, args=self.args)
            vae.buildModel()
            vae.optimizeModel()

        elif self.args.model == Strings.STIB_WO_IR:

            stibWoReg = STIBwoIRModel(dataset=self.dataset, z0_size=2, z1_size=1, y_size=2, x_size=2, args=self.args)
            stibWoReg.buildModel()
            stibWoReg.optimizeModel()

        elif self.args.model == Strings.STIB:

            stib = STIBModel(dataset=self.dataset, z0_size=2, z1_size=1, y_size=2, x_size=2, args=self.args)
            stib.buildModel()
            stib.optimizeModel()

        elif self.args.model == Strings.CVAE:

            cvae = CVAEModel(dataset=self.dataset, z0_size=2, z1_size=1, y_size=2, x_size=2, args=self.args)
            cvae.buildModel()
            cvae.optimizeModel()

        elif self.args.model == Strings.CVIB:

            cvib = CVIBModel(dataset=self.dataset, z0_size=2, z1_size=1, y_size=2, x_size=2, args=self.args)
            cvib.buildModel()
            cvib.optimizeModel()

        else:
            self.log.error("Model to train not found!")
