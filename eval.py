import argparse
import torch
from data.utils import DataSet
from data.utils import NormalizeToImage
from torch.utils.data import DataLoader

from model import Generator
from trainer import Predictor

parser = argparse.ArgumentParser(prog = 'top', description = 'Evaluate Super-Resolution GAN')

parser.add_argument('-m', '--model', type=str, default='model/generator.pt',
                   help='Checkpoint directory of pretrained Generator model')
parser.add_argument('-d', '--dir', type=str, default='test',
                   help='Directory of test images')
parser.add_argument('-is', '--input_size', type=int, default=24,
                    help='Input image will be upscaled in *4 times')

args = parser.parse_args()

generator = Generator(in_channels=3, hid_channels=64, out_channels=3)
generator.load_state_dict(torch.load(args.model))

predictor = Predictor(model=generator, transform=NormalizeToImage)

dataset = DataSet(path=args.dir, input_size=(args.input_size, args.input_size))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

if __name__ == "__main__":
    predictor.predict(dataloader, path=args.dir)