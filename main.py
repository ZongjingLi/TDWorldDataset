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
from scene_distribution import SceneController, indoor_options
print(indoor_options)
def generate(args):
    split = args.split
    scene_num = args.scene_num
    scene_controller = SceneController(split = args.split, output_directory = args.output_directory)
    room_split = 0
    for k in range(len(indoor_options)):
        scene_name = indoor_options[k]
        for room_name in list(scene_name.values())[0]:
            for num in range(scene_num):
                scene_info = scene_controller.generate_scene(img_name = f"{split}_{num + scene_num*room_split}", scene_name = room_name)
                vqa_info = scene_controller.generate_vqa_pairs(scene_info, args.vqa_pair_per_scene)
                scene_controller.reset_scene()
                sys.stdout.write(f"\rk:{k} generating scenes: [{num+1} / {scene_num }]")
            room_split += 1
    scene_controller.communicate({"$type": "terminate"})
    sys.stdout.write(f"\ngenerate split:{split} k:{k} num:{scene_num} scenes done.")

local = True
dataset_dir = "/Users/melkor/Documents/datasets" if local else "datasets"


parser = argparse.ArgumentParser()
parser.add_argument("--scene_num",                              default = 30)
parser.add_argument("--output_directory",                       default = f"{dataset_dir}/TDWRoom")
parser.add_argument("--split",                                  default = "train")
parser.add_argument("--vqa_pair_per_scene",                     default = 2)

if __name__ == "__main__":
    args = parser.parse_args()

    generate(args)