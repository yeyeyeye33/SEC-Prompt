import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/inr.json',
                        help='Json file of settings.')
    parser.add_argument('--lamda', type=float, default=0.5,help='lamda of Discriminative Prompt Clustering Loss.')
    parser.add_argument('--aug', type=int, default=2, help='n of data augmentation.')
    parser.add_argument('--prompt_token_num', type=int, default=10, help='prompt_token_num*2.')
    parser.add_argument('--prompt_pool_num', type=float, default=0.5, help='prompt pool size：class_num*prompt_pool_num')
    return parser

if __name__ == '__main__':
    main()
