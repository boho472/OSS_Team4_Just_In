import argparse
import sys
import os
from tracking_main import tracking_val_seq


parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--cfg_file', type=str, default="src/configs/tracking.yaml",
                    help='specify the config for tracking')
local_args, remaining_args = parser.parse_known_args()
sys.argv = [sys.argv[0]]
tracking_val_seq(local_args)