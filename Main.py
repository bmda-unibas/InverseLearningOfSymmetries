import argparse
import logging
import sys

import Strings
from Dataset import Dataset
from Evaluation import Evaluation
from Training import Training

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if __name__ == "__main__":
    # set logger
    log = logging.getLogger(__name__)
    logging.info("NeurIPS 2020 Implementation our paper: Inverse Learning of Symmetries")

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='Set training or evaluation mode', default="evaluate")
    parser.add_argument('--model', help='Set model', default="CVAE")
    parser.add_argument('--pretrained', help='Set pretrained model', default="pretrained/CVIB.ckpt")
    parser.add_argument('--save_path', help='Set save path model', default="saved_models/STIB.ckpt")
    parser.add_argument('--eval_dataset', help='Set evaluation dataset', default="dataset/testset.pickle")
    parser.add_argument('--batch_size', help='Set batch size', default=60)
    parser.add_argument('--plot_path', help='Set plot path', default="plots/")
    parser.add_argument('--iterations', help='Set number of iterations', default=150000)
    args = parser.parse_args()

    # load dataset
    dataset = Dataset(args.eval_dataset)

    # training
    if args.mode == Strings.TRAIN:
        training = Training(args=args)
        training.train()

    # evaluation
    if args.mode == Strings.EVALUATE:
        evaluation = Evaluation(args=args, dataset=dataset)
        evaluation.evaluate()
