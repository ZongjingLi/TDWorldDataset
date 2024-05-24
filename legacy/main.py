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

local = False
dataset_dir = "/Users/melkor/Documents/datasets" if local else "datasets"
dataset_name = "TDWRoom"

parser = argparse.ArgumentParser()
parser.add_argument("--task",                                   default = "object_room")
parser.add_argument("--scene_num",                              default = 500)
parser.add_argument("--output_directory",                       default = f"{dataset_dir}/{dataset_name}")
parser.add_argument("--split",                                  default = "test")
parser.add_argument("--vqa_pair_per_scene",                     default = 2)


valid_tasks = [
    "object_room", "multiview_scene"
]

if __name__ == "__main__":
    args = parser.parse_args()
    if args.task not in valid_tasks:
        sys.stdout.write(f"\rTask:{args.task} is not found in the valid tasks. exiting...")
        sys.exit()
    if args.task == "object_room":
        print("start to generate the object room")
        print("object room generation completed.")