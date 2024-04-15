'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-15 10:11:27
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-15 10:11:28
 # @ Description: This file is distributed under the MIT license.
'''
import sys
import os
import argparse
from scene_distribution import SceneController

def generate(args):
    split = args.split
    scene_num = args.scene_num
    scene_controller = SceneController(args.split, args.output_directory)
    for num in range(scene_num):
        scene_controller.generate_scene(img_name = f"{split}_{num}")
        scene_controller.reset_scene()
        sys.stdout.write(f"\n{num+1} / {scene_num}")
    ys.stdout.write(f"\ngenerate split:{split} num:{scene_num} scenes done.")

parser = argparse.ArgumentParser()
parser.add_argument("--scene_num",                              default = 1)
parser.add_argument("--output_directory",                       default = "datasets/TDWRoom")
parser.add_argument("--split",                                  default = "train")
parser.add_argument("--vqa_pair_per_scene",                     default = 2)

if __name__ == "__main__":
    args = parser.parse_args()

    generate(args)