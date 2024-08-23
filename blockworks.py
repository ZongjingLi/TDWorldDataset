'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-08-20 09:03:16
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-08-20 09:03:22
 # @ Description: This file is distributed under the MIT license.
'''

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.robot import Robot
from tdw.add_ons.mouse import Mouse
from tdw.add_ons.keyboard import Keyboard
from tdw.output_data import OutputData, Images, Transforms, Rigidbodies, Bounds, SegmentationColors, IdPassSegmentationColors
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.output_data import OutputData, Images
from utils import get_material, get_model

import os
import random
import math
import random
import numpy as np

from typing import List

import torch
import matplotlib.pyplot as plt
import pandas as pd

from rinarak.utils.os import save_json, load_json

random_seed = 11

class BlockWorksController(Controller):
    def __init__(self,
                 split = "train",
                 resolution : tuple[int, int] = (512,512),
                 output_directory = "datasets/{}",
                 name = "Plagueworks",
                 counter = 0,
                 load_scene = None,
                 port = 1932):
        super().__init__(port = port)
        self.room_name = "box_room_2018"
        if load_scene is not None:
            if isinstance(load_scene, str): load_scene = load_json(load_scene)
            assert isinstance(load_scene, dict), "input load scene is not a valid"
            if "room_name" in load_scene: self.room_name = load_scene["room_name"]
        self.W, self.H = resolution
        self.moveables = []
        self.immoveables = []
        self.split = split
        self.output_directory = output_directory.format(name) + f"/{split}"

        """create the camera at the default location"""
        self.camera_locations = []
        self.camera_lookats = []
        if "camera" in load_scene:
            cameras = []
        else:
            self.camera_location = {"x" : .75, "y": .5, "z" : .75}
            self.camera_lookat = {"x" : 0, "y": .3, "z" : 0}
            camera = ThirdPersonCamera(position=self.camera_location,
                           look_at=self.camera_lookat,
                           avatar_id="a")
            self.camera_locations.append(self.camera_location)
            self.camera_lookats.append(self.camera_lookat)
            cameras = [camera]
        self.camera = camera
        self.cameras = cameras # create avatars
        self.mouse = Mouse(avatar_id="a") # create a keyboard to control 
        self.keyboard = Keyboard()

        """keep track of objects generated in the scene with: id, model, texture, color etc"""
        self.object_ids = []
        self.object_infos = {}
        self.om= ObjectManager(transforms=False, bounds=True, rigidbodies=True)
        self.rng: np.random.RandomState = np.random.RandomState(32)


        """add realistic shading to the environment"""
        #self.interior_scene_lighting = InteriorSceneLighting(rng=np.random.RandomState(random_seed))
        #self.interior_scene_lighting,

        """get all the add-ons to the environment"""
        self.communicate(self.get_add_scene(scene_name=self.room_name))
        self.add_ons.extend([self.camera, self.keyboard, self.mouse])

        """create the default room setup, set the resolution etc"""
        self.basic_setup()
        self.counter = counter

        if load_scene is not None:
            assert isinstance(load_scene, dict), "input load scene is not a valid"
            objects = load_scene["objects"]
            
            for object_id in objects:
                tdw_object = objects[object_id]
                color = tdw_object["color"] if "color" in tdw_object else None

                self.add_object(
                    tdw_object["model"],
                    tdw_object["position"],
                    tdw_object["rotation"],
                    color = color,
                    id = int(object_id)
                    )



                """
                s = 0.1
                self.communicate(self.get_add_object(model_name=tdw_object["model"],
                                                              object_id=int(object_id),
                                                              library="models_flex.json",
                                                              position=tdw_object["position"],
                                                              rotation = tdw_object["rotation"],
                                                              kinematic=True, mass = 1.0) )
                self.object_ids.append(int(object_id))
                self.communicate({"$type": "set_color",
                                 "id":int(object_id),
                                 "color": color})
                """


        self.scene_save_path = self.output_directory + "/scene_setup.json"

        print("PlageWorks environment is created, all objects loaded.")
    
    def add_camera(self,camera_location = None, camera_lookat = None, id = "avatar"):
        camera_location = {"x": .8, "y":.4, "z":-.7}
        camera_lookat = {"x": 0.0, "y":0.3, "z":-0.0}
        add_camera = ThirdPersonCamera(position = camera_location,
                           look_at = camera_lookat,
                           avatar_id=id)
        self.add_ons.extend([add_camera])
        self.cameras.append(add_camera)
        self.camera_locations.append(camera_location)
        self.camera_lookats.append(camera_lookat)
    
    def save_scene_setup(self, path = None):
        if path is None: path = self.scene_save_path
        scene_setup = {}
        object_infos = []
        for object_id in self.object_ids:
            object_infos[object_id] = {
                "model": "vase_01",
                "position": {"x":0,"y":0,"z":0},
                "rotation": {"x":.7,"y":0,"z":.7},
            }
        scene_setup["objects"] = object_infos
        save_json(scene_setup, path)

    def basic_setup(self):
        commands = []
        commands.extend([
        {"$type": "set_screen_size", "width":self.W, "height": self.H},
        ])
        responds = self.communicate(commands)
        #
        self.setup_keyboard()
    
    def setup_keyboard(self):
        self.keyboard.listen(key="Escape", commands = [{"$type": "terminate"}])
        self.keyboard.listen(key="C", function = self.capture)
        self.keyboard.listen(key="G", function = self.capture_sequence)
        self.keyboard.listen(key="F", function = self.apply_force)
        self.keyboard.listen(key="M", function = self.capture_multiview)

    def apply_force(self):
        commands = []
        commands.append({"$type": "apply_force_to_object",
                 "id": self.object_ids[np.random.randint(0, len(self.object_ids))],
                 "force": {"x": 0.5, "y": 0.3, "z": 0.}})

        self.communicate(commands)
        
    
    def capture_sequence(self, roll_out = 1):
        self.capture(f"{self.counter}_1")
        self.apply_force()
        for i in range(roll_out):
            self.communicate([])
        self.capture(f"{self.counter}_2")
        self.counter += 1

        return

    def capture_multiview(self, view_num = 5, save_name = None):
        save_name = self.counter if save_name is None else save_name
        img_name = f"{save_name}"
        commands = [
             {"$type": "set_pass_masks", "pass_masks": ["_img", "_id", "_albedo"], "avatar_id": f"{camera.avatar_id}"} for camera in self.cameras
        ]
        commands.extend([
        {"$type": "send_images", "frequency": "always", "ids": [camera.avatar_id for camera in self.cameras]}])

        """give out the color and id of the objects in the image"""
        commands.extend([
            {"$type": "send_segmentation_colors",
            "frequency": "once"},
            {"$type": "send_id_pass_segmentation_colors",
            "frequency": "always"}])
        responds = self.communicate(commands)

        save_dir = self.output_directory+ f"/img/{img_name}"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        for i in range(len(responds)):
            r_id = OutputData.get_data_type_id(responds[i])
            if r_id == "imag":
                image = Images(responds[i])
                avatar_id = image.get_avatar_id()
                TDWUtils.save_images(image, filename = f"{avatar_id}", output_directory = self.output_directory+ f"/img/{img_name}/")
        
        """save the camera information of the mulit view from different avatars"""
        camera_info = {
            "location": self.camera_locations,
            "lookat": self.camera_lookats,
            "names": [camera.avatar_id for camera in self.cameras]
        }
        save_json(camera_info, self.output_directory + f"/scene/camera_{img_name}.json")
        self.counter += 1

    def capture(self, save_name = None):
        #print(controller.om.categories[self.object_ids[-1]])
        save_name = self.counter if save_name is None else save_name
        img_name = f"{save_name}"
        commands = [
             {"$type": "set_pass_masks", "pass_masks": ["_img", "_id", "_albedo"], "avatar_id": f"{camera.avatar_id}"} for camera in self.cameras
        ]
        commands.extend([
        {"$type": "send_images", "frequency": "always", "ids": [camera.avatar_id for camera in self.cameras]}])

        """give out the color and id of the objects in the image"""
        commands.extend([
            {"$type": "send_segmentation_colors",
            "frequency": "once"},
            {"$type": "send_id_pass_segmentation_colors",
            "frequency": "always"}])
        responds = self.communicate(commands)

        segmentation_colors_per_object = dict()
        segmentation_colors_in_image = list()
        binary_mask = torch.zeros([self.W, self.H])

        object_id_sequence = []
        for i in range(len(responds)):
            r_id = OutputData.get_data_type_id(responds[i])
            if r_id == "imag":
                image = Images(responds[i])
                avatar_id = image.get_avatar_id()
                TDWUtils.save_images(image, filename = f"{img_name}_{avatar_id}", output_directory = self.output_directory+ f"/img")

        id_map = (torch.tensor(plt.imread(self.output_directory + f"/img/id_{img_name}_{avatar_id}.png")) * 255).int()

        for i in range(len(responds)):
            r_id = OutputData.get_data_type_id(responds[i])
            if r_id == "segm":
                #segm = IdPassSegmentationColors(responds[i])
                segm = SegmentationColors(responds[i])
                object_id_sequence = []
                obj_counter = 0
                for j in range(segm.get_num()):
                    object_id = segm.get_object_id(j)
                    if (object_id in self.object_ids):
                        segmentation_color = segm.get_object_color(j)
                        segmentation_colors_per_object[object_id] = segmentation_color

                        locs = torch.max(id_map == torch.tensor(segmentation_color), dim = - 1, keepdim = False).values

                        binary_mask[locs] = obj_counter
                        obj_counter += 1
                        object_id_sequence.append(object_id)

                    else:
                        print("not found:",object_id)

        np.save(self.output_directory + f"/img/mask_{img_name}" ,binary_mask)
        np.save(self.output_directory + f"/scene/ids_{img_name}", object_id_sequence)

        camera_info = {
            "location": self.camera_locations,
            "lookat": self.camera_lookats,
            "names": [camera.avatar_id for camera in self.cameras]
        }
        save_json(camera_info, self.output_directory + f"/scene/camera_{img_name}.json")

        print(f"done:{img_name}")

    def add_object(self, 
                   model_name : str,
                   position : dict[str, float] = {"x": 0, "y":0.0, "z":0},
                   rotation : dict[str, float] = {"x": 0, "y":0, "z":0},
                   scale : float = 0.1, color = None, id = None):
        """
        Args:
            model_name : a string corresponding to the model used to create
        Returns:
            return the object created with model_name, position, rotation etc
        """
        object_id = self.get_unique_id() if id is None else int(id)

        self.communicate([self.get_add_object(
            object_id = object_id,library="models_flex.json",
            model_name = model_name, position = position, rotation = rotation,
        )])

        self.communicate([
            {"$type": "scale_object", "id": object_id, "scale_factor": {"x": scale, "y": scale, "z": scale}}
        ])
    
        
        self.object_ids.append(object_id)
        self.object_infos[object_id] = {
            "model": model_name,
            "position": position,
            "rotation": rotation,
            "scale" : scale,
        }

        if color is not None:
            color_ = {"r": 1.0, "g": 0, "b": 0, "a": 1.0}
            self.communicate([
                {"$type": "set_color",
                "color": color,
                "id": object_id}
            ])
            self.object_infos[object_id]["color"] = color
        print(f"add:{model_name}")
        return

    def replace_equivalence(self, equivalence_table):
        for obj_id in self.object_ids:
            model_name = self
        return

    def run(self):
        done = False
        print("start running")
        while not done:
            self.communicate([])
        self.communicate({"$type": "terminate"})

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name",           default = "Blockworks")
parser.add_argument("--split",                  default = "train")
parser.add_argument("--dataset_dir",            default =  "/Users/melkor/Documents/datasets/{}")
args = parser.parse_args()



if __name__ == "__main__":
    
    dataset_name = args.dataset_name
    dataset_name = "Blockworks"
    split = args.split
    dataset_dir = args.dataset_dir
    controller = BlockWorksController(
        split = split,
        name = dataset_name,
        output_directory = dataset_dir,
        load_scene = dataset_dir.format(dataset_name) + f"/{split}/scene_setup.json")

    #controller.add_camera()
    controller.run()
