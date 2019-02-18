from dl.util import load_gin_configs
from dl.algorithms import SAC
import argparse, os

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train SAC.')
    parser.add_argument('logdir', type=str, help='logdir')
    parser.add_argument('-c', '--gin_config', type=str, help='gin config')
    parser.add_argument('-b', '--gin_bindings', nargs='+', help='gin bindings to overwrite config')
    args = parser.parse_args()
    if args.gin_config is None:
        config = os.path.dirname(os.path.dirname(__file__)) + '/configs/sac.gin'
    else:
        config = args.gin_config
    load_gin_configs([config], args.gin_bindings)

    sac = SAC(args.logdir)
    sac.train()
