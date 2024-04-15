'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-15 10:11:27
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-15 10:11:28
 # @ Description: This file is distributed under the MIT license.
'''

import argparse

def generate(args):
    return 

parser = argparse.ArgumentParser()
parser.add_argument("--scene_num",                              default = 4)
parser.add_argument("--output_directory",                       default = "datasets/TDWRoom")
parser.add_argument("--split",                                  default = "train")
parser.add_argument("--vqa_pair_per_scene",                     default = 2)

if __name__ == "__main__":
    args = parser.parse_args()

    generate(args)