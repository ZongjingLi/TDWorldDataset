'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-05-23 11:39:17
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-05-23 11:39:18
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

class PlagueWorksController(Controller):
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
        self.W, self.H = resolution
        self.moveables = []
        self.immoveables = []
        self.split = split
        self.output_directory = output_directory.format(name) + f"/{split}"

        """create the camera at the default location"""
        self.camera_location = {"x" : 1., "y": 1., "z" : 1.}
        self.camera_lookat = {"x" : 0, "y": 0, "z" : 0}
        camera = ThirdPersonCamera(position=self.camera_location,
                           look_at=self.camera_lookat,
                           avatar_id="a")
        self.camera = camera
        self.mouse = Mouse(avatar_id="a") # create a keyboard to control 
        self.keyboard = Keyboard()

        """keep track of objects generated in the scene with: id, model, texture, color etc"""
        self.object_ids = []
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
            assert isinstance(load_scene, dict), "input load scene is not a valid "

        print("PlageWorks environment is created, all objects loaded.")

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

    def apply_force(self):
        commands = []
        commands.append({"$type": "apply_force_to_object",
                 "id": self.object_ids[-1],
                 "force": {"x": -1.8, "y": 20., "z": 1.5}})
        self.communicate(commands)
    
    def capture_sequence(self, roll_out = 3):
        self.capture(f"{self.counter}_1")
        self.apply_force()
        for i in range(roll_out):
            self.communicate([])
        self.capture(f"{self.counter}_2")
        self.counter += 1
        return

    def capture_multiview(self, view_num = 5):
        pass

    def capture(self, save_name = None):
        #print(controller.om.categories[self.object_ids[-1]])
        save_name = self.counter if save_name is None else save_name
        img_name = f"{save_name}"
        commands = []
        commands.extend([
        {"$type": "set_pass_masks", "pass_masks": ["_img", "_id", "_albedo"], "avatar_id": "a"},
        {"$type": "send_images", "frequency": "always", "ids": ["a"]}])

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
                TDWUtils.save_images(image, filename = f"{img_name}", output_directory = self.output_directory+ f"/img")

        id_map = (torch.tensor(plt.imread(self.output_directory + f"/img/id_{img_name}.png")) * 255).int()

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
        print(object_id_sequence)

        print(f"done:{img_name}")

    def add_object(self, 
                   model_name : str,
                   position : dict[str, float] = {"x": 0, "y":0.0, "z":0},
                   rotation : dict[str, float] = {"x": 0, "y":0, "z":0},
                   scale : float = 1.0, id = None):
        """
        Args:
            model_name : a string corresponding to the model used to create
        Returns:
            return the object created with model_name, position, rotation etc
        """
        object_id = self.get_unique_id() if id is None else id
        self.communicate([self.get_add_object(
            object_id = object_id,
            model_name = model_name, position = position, rotation = rotation
        )])
        self.object_ids.append(object_id)
        print(f"add:{model_name}")
        return

    def run(self):
        done = False
        print("start running")
        while not done:
            self.communicate([])
        self.communicate({"$type": "terminate"})

if __name__ == "__main__":
    dataset_dir = "/Users/melkor/Documents/datasets/{}"
    controller = PlagueWorksController(split = "train", name = "Plagueworks", output_directory = dataset_dir, load_scene = None)
    #controller.capture()
    controller.add_object("vase_01")
    controller.add_object("vase_05", position={"x":0.5, "y":0.0, "z":0.2})
    controller.run()
   
    #controller.capture()