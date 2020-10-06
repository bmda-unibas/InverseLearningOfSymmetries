import logging


class Training:
    def __init__(self):
        self.log = logging.getLogger(__name__)

    def train(self):
        self.log.info("Training started ...")
