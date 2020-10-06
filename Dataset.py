import logging
import pickle
import sys

import numpy as np

np.random.seed(1234)


class Dataset:
    def __init__(self, path_to_eval_dataset):
        self.log = logging.getLogger(__name__)
        self.log.info("Load dataset ...")
        self.path_to_eval_dataset = path_to_eval_dataset
        # load evaluation dataset
        self.loadEvalDataset()

    def loadEvalDataset(self):
        try:
            filehandler = open(self.path_to_eval_dataset, "rb")
            test_list = pickle.load(filehandler)
            filehandler.close()

            self.x = test_list[0]
            self.y = test_list[1]
            self.t = test_list[2]

            t_min = min(self.t)
            t_max = max(self.t)
            delta = (t_max - t_min) / 10.0

            array = np.zeros(10)

            for i in range(10):
                array[i] = t_min
                t_min = t_min + delta

            self.bins = np.digitize(self.t, array)
        except:
            self.log.error("Set correct dataset ...")
            sys.exit()

    def simData(self, n=300, d=2):
        x = 8 * np.random.uniform(0, 1, n * d) - 4
        x = np.reshape(x, (n, d))

        t1 = 2 * x[:, 0] + 1 * x[:, 1] + 1e-1 * np.random.normal(0, 1, n)
        t2 = t1

        y = np.zeros((n, d))

        b = 10
        y[:, 0] = 5e-2 * (t1 + b * t2) * np.cos(t2) + 2e-1 * np.random.normal(0, 1, n)
        y[:, 1] = 5e-2 * (t1 + b * t2) * np.sin(t2) + 2e-1 * np.random.normal(0, 1, n)
        return x, y, t1

    def next_batch(self, batch_size):
        x, y, t = self.simData(n=batch_size)
        return x, y, t

    def get_eval_data(self):
        return self.x, self.y, self.t, self.bins
