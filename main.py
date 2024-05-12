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
from scene_distribution import SceneController, KitchenController, indoor_options

def generate(args):
    split = args.split
    scene_num = args.scene_num
    scene_controller = SceneController(split = args.split, output_directory = args.output_directory)
    room_split = 0
    for k in range(1):
        scene_name = indoor_options[k]
        for room_name in [1]:
            for num in range(scene_num):
                scene_info = scene_controller.generate_scene(img_name = f"{num + scene_num*room_split}", scene_name = room_name)
                vqa_info = scene_controller.generate_vqa_pairs(scene_info, args.vqa_pair_per_scene)
                scene_controller.reset_scene()
                sys.stdout.write(f"\rk:{k} generating scenes: [{num+1} / {scene_num }]")
            room_split += 1
    scene_controller.communicate({"$type": "terminate"})
    sys.stdout.write(f"\ngenerate split:{split} k:{k} num:{scene_num} scenes done.")

def generate_kitchen(args, split_name = "kllk"):
    split = args.split
    scene_num = args.scene_num
    scene_controller = KitchenController(split = args.split, output_directory = args.output_directory)
    for num in range(args.scene_num):
        scene_controller.generate_scene(str(num))
        scene_controller.reset_scene()

local = True
dataset_dir = "/Users/melkor/Documents/datasets" if local else "datasets"
dataset_name = "TDWRoom"

parser = argparse.ArgumentParser()
parser.add_argument("--scene_num",                              default = 500)
parser.add_argument("--output_directory",                       default = f"{dataset_dir}/{dataset_name}")
parser.add_argument("--split",                                  default = "test")
parser.add_argument("--vqa_pair_per_scene",                     default = 2)

if __name__ == "__main__":
    args = parser.parse_args()
    if dataset_name == "TDWRoom":
        generate(args)
    if dataset_name == "TDWKitchen":
        generate_kitchen(args)