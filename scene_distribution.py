'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-14 10:42:52
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-14 10:42:59
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
from tdw.output_data import OutputData, Images
from utils import get_material, get_model

import math
import random
import numpy as np

from typing import List

import torch
import matplotlib.pyplot as plt
import pandas as pd

indoor_options = [
    {"object_room": ["box_room_2018"]}, 
    {"kitchen": ["mm_kitchen_1a", "mm_kitchen_1b", "mm_kitchen_2a", "mm_kitchen_2b"]},
    {"craftroom": ["mm_craftroom_4a",  "mm_craftroom_1a" ,"monkey_physics_room"]}, 
    {"diningroom": ["mm_craftroom_4a",  "mm_craftroom_1a" ]}
]

class SceneController(Controller):
    def __init__(self, split = "train", port = 1928, output_directory = "datasets/TDWRoom"):
        super().__init__(port = port)
        self.moveables = []
        self.immoveables = []
        self.split = split
        self.output_directory = output_directory + f"/{split}"

        """keep track of objects generated in the scene with: id, model, texture, color etc"""
        self.object_ids = []
        self.om= ObjectManager(transforms=False, bounds=True, rigidbodies=True)
        self.rng: np.random.RandomState = np.random.RandomState(32)

        self.W, self.H = 512, 512


    def reset_scene(self, option = None):
        for object_id in self.object_ids:
            self.communicate({"$type": "destroy_object", "id": object_id})
        self.object_ids = []

    def set_floor(self) -> List[dict]:
        materials = ["parquet_wood_mahogany", "parquet_long_horizontal_clean", "parquet_wood_red_cedar"]
        material_name = materials[self.rng.randint(0, len(materials))]
        texture_scale: float = float(self.rng.uniform(4, 4.5))
        return [self.get_add_material(material_name=material_name),
                {"$type": "set_floor_material",
                 "name": material_name},
                {"$type": "set_floor_texture_scale",
                 "scale": {"x": texture_scale, "y": texture_scale}},
                {"$type": "set_floor_color",
                 "color": {"r": float(self.rng.uniform(0.7, 1)),
                           "g": float(self.rng.uniform(0.7, 1)),
                           "b": float(self.rng.uniform(0.7, 1)),
                           "a": 1.0}}]

    def set_walls(self) -> List[dict]:
        materials = ["cinderblock_wall", "concrete_tiles_linear_grey", "old_limestone_wall_reinforced"]
        material_name = materials[self.rng.randint(0, len(materials))]
        texture_scale: float = float(self.rng.uniform(0.1, 0.3))
        return [self.get_add_material(material_name=material_name),
                {"$type": "set_proc_gen_walls_material",
                 "name": material_name},
                {"$type": "set_proc_gen_walls_texture_scale",
                 "scale": {"x": texture_scale, "y": texture_scale}},
                {"$type": "set_proc_gen_walls_color",
                 "color": {"r": float(self.rng.uniform(0.7, 1)),
                           "g": float(self.rng.uniform(0.7, 1)),
                           "b": float(self.rng.uniform(0.7, 1)),
                           "a": 1.0}}]

    def set_ceiling(self) -> List[dict]:
        materials = ["bricks_red_regular", "bricks_chatham_gray_used", "bricks_salem_matt_used"]
        material_name = materials[self.rng.randint(0, len(materials))]
        texture_scale: float = float(self.rng.uniform(0.1, 0.2))
        return [{"$type": "create_proc_gen_ceiling"},
                self.get_add_material(material_name=material_name),
                {"$type": "set_proc_gen_ceiling_material",
                 "name": material_name},
                {"$type": "set_proc_gen_ceiling_texture_scale",
                 "scale": {"x": texture_scale, "y": texture_scale}},
                {"$type": "set_proc_gen_ceiling_color",
                 "color": {"r": float(self.rng.uniform(0.7, 1)),
                           "g": float(self.rng.uniform(0.7, 1)),
                           "b": float(self.rng.uniform(0.7, 1)),
                           "a": 1.0}}]
    def add_object(self, model, position = {"x": -0.1, "y": 0, "z": -0.8}):
        object_id = self.get_unique_id()
        material_record = get_material("parquet_long_horizontal_clean")
        model_record = get_model(model)
        self.communicate([
            {"$type": "add_object",
                "name": model_record.name,
                "url": model_record.get_url(),
                "scale_factor": model_record.scale_factor,
                "position": position,
                "rotation": {"x": 0, "y": 0, "z": 0},
                "category": model_record.wcategory,
                "id": object_id},
        ])
        """
        self.communicate([
            {"$type": "set_visual_material",
                "material_index": 0,
                "material_name": material_record.name,
                "object_name": "Object017",
                "id": object_id}
        ])
        """
        self.object_ids.append(object_id)
    
    def scene(self):
        return

    def generate_scene_kitchen_tdworld(self,model_name = None, height = None, img_name = 0, scene_name = None, numberw=3):
        """
        generate a scene with room tdw_room, randomly choose objects that are moveable and put it on the floor.
        """
        

    def generate_scene_moveable_tdworld(self,model_name = None, height = None, img_name = 0, scene_name = None, numberw=3):
        """
        generate a scene with room tdw_room, randomly choose objects that are moveable and put it on the floor.
        """
        W, H = self.W, self.H
        if model_name is None: model_name = "iron_box"
        if height is None: height = 0.1 + random.random() * 0.3
        df = pd.read_csv('metadata/ObjectDict.csv')

# 选择第二项为"y"的行，并提取第一项作为数组
        filtered_array = df.loc[df['movable'] == 'y', 'object'].values
        n = numberw  # 选择要打印的元素数量
        random_elements = random.sample(list(filtered_array), n)
        print("_________________________________________________________")
        print(random_elements)
# 分别打印随机选择的元素
        #for element in random_elements:
            #print(element)
# 打印数组
        #print(filtered_array) 

        commands = []
        """create an avatar to observe the environment"""
        theta = np.random.random() * 2 * np.pi
        scale = np.random.random() * 0.2 + 1
        commands.extend(TDWUtils.create_avatar(position={"x": np.cos(theta) * scale, "y": 2, "z": np.sin(theta) * scale},
                                       avatar_id="a",
                                       look_at={"x": 0.0, "y": 0.2, "z": 0.0}))
        
        """create the room and setup the floor, walls etc to make it look real haha"""
        self.communicate(TDWUtils.create_empty_room(12,12))
        self.communicate(self.set_floor())
        self.communicate(self.set_walls())

        #if scene_name is None: scene_name = "mm_craftroom_4a"
        #self.communicate(self.get_add_scene(scene_name=scene_name))

        """add some objects in the scene"""
        for element in random_elements:

            self.add_object(element, position = {"x":random.uniform(-0.5, 0.5), "y":height, "z":random.uniform(-0.5, 0.5)})

        #self.add_obj1_on_obj2("small_table_green_marble", "jug01")
        print(self.object_ids)
        #self.add_object("white_lounger_chair")
    
        
        commands.extend([
            {"$type": "set_screen_size", "width":W, "height": H},
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
        
        binary_mask = torch.zeros([W, H])

        for i in range(len(responds)):
            r_id = OutputData.get_data_type_id(responds[i])
            if r_id == "imag":
                image = Images(responds[i])
                avatar_id = image.get_avatar_id()
                TDWUtils.save_images(image, filename = f"{img_name}", output_directory = self.output_directory+ f"/img")

        image = (torch.tensor(plt.imread(self.output_directory + f"/img/id_{img_name}.png")) * 255).int()
        
        for i in range(len(responds)):
            r_id = OutputData.get_data_type_id(responds[i])
            if r_id == "segm":
                segm = SegmentationColors(responds[i])
                for j in range(segm.get_num()):
                    object_id = segm.get_object_id(j)
                    print(object_id)
                    if (object_id in self.obect_ids):
                        #object_name = self.object_ids.index(object_id)
                        segmentation_color = segm.get_object_color(j)
                        segmentation_colors_per_object[object_id] = segmentation_color
                        #print(self.object_ids.index(object_id), segmentation_color)
                        #print((torch.tensor(image[:,:]) == segmentation_color).shape)
                        locs = torch.max(image == torch.tensor(segmentation_color), dim = - 1, keepdim = False).values
                     #print(locs.shape)
                     #print(binary_mask.shape)
                        binary_mask[locs] = self.object_ids.index(object_id) + 1
                    """
                    for i in range(self.W):
                        for j in range(self.H):
                            #print(image[i,j] ,torch.tensor(segmentation_color),list(image[i,j]) == list(torch.tensor(segmentation_color).int()))
                            if list(image[i,j]) == list(torch.tensor(segmentation_color).int()):
                                binary_mask[i,j] = self.object_ids.index(object_id)
                    """
            #np.save(binary_mask,)
        np.save(self.output_directory+ f"/img/mask_{img_name}" ,binary_mask)
    
        """
            elif r_id == "ipsc":
                ipsc = IdPassSegmentationColors(responds[i])
                for j in range(ipsc.get_num_segmentation_colors()):
                    print(ipsc.get_segmentation_color(j))
                    segmentation_colors_in_image.append(ipsc.get_segmentation_color(j))
        """
        for object_id in segmentation_colors_per_object:
            for i in range(len(segmentation_colors_in_image)):
                if any((segmentation_colors_in_image[i] == j).all() for j in segmentation_colors_per_object.values()):
                    #print(object_id, segmentation_color[i])
                    break

        scene_info = {}
        return scene_info





    def generate_scene(self,model_name = None, height = None, img_name = 0, scene_name = None):
        W, H = self.W, self.H
        if model_name is None: model_name = "iron_box"
        if height is None: height = 0.8 + random.random() * 0.3
        

        commands = []
        """create an avatar to observe the environment"""
        theta = np.random.random() * 2 * np.pi
        scale = np.random.random() * 0.2 + 1.8
        commands.extend(TDWUtils.create_avatar(position={"x": np.cos(theta) * scale, "y": 1.32, "z": np.sin(theta) * scale},
                                       avatar_id="a",
                                       look_at={"x": 0.0, "y": 0.4, "z": 0.0}))
        
        """create the room and setup the floor, walls etc to make it look real haha"""

        #self.communicate(self.set_floor())
        #self.communicate(self.set_walls())

        if scene_name is None: scene_name = "mm_craftroom_4a"
        self.communicate(self.get_add_scene(scene_name=scene_name))

        """add some objects in the scene"""

        self.add_object(model_name, position = {"x":0, "y":height, "z":0})

        self.add_obj1_on_obj2("small_table_green_marble", "jug01")

        self.add_object("white_lounger_chair")
    
        
        commands.extend([
            {"$type": "set_screen_size", "width":W, "height": H},
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
        
        binary_mask = torch.zeros([W, H])

        for i in range(len(responds)):
            r_id = OutputData.get_data_type_id(responds[i])
            if r_id == "imag":
                image = Images(responds[i])
                avatar_id = image.get_avatar_id()
                TDWUtils.save_images(image, filename = f"{img_name}", output_directory = self.output_directory+ f"/img")

        image = (torch.tensor(plt.imread(self.output_directory + f"/img/id_{img_name}.png")) * 255).int()
        
        for i in range(len(responds)):
            r_id = OutputData.get_data_type_id(responds[i])
            if r_id == "segm":
                segm = SegmentationColors(responds[i])
                for j in range(segm.get_num()):
                    object_id = segm.get_object_id(j)
                    object_name = self.object_ids.index(object_id)
                    segmentation_color = segm.get_object_color(j)
                    segmentation_colors_per_object[object_id] = segmentation_color
                    #print(self.object_ids.index(object_id), segmentation_color)
                    #print((torch.tensor(image[:,:]) == segmentation_color).shape)
                    locs = torch.max(image == torch.tensor(segmentation_color), dim = - 1, keepdim = False).values
                    #print(locs.shape)
                    #print(binary_mask.shape)
                    binary_mask[locs] = self.object_ids.index(object_id) + 1
                    """
                    for i in range(self.W):
                        for j in range(self.H):
                            #print(image[i,j] ,torch.tensor(segmentation_color),list(image[i,j]) == list(torch.tensor(segmentation_color).int()))
                            if list(image[i,j]) == list(torch.tensor(segmentation_color).int()):
                                binary_mask[i,j] = self.object_ids.index(object_id)
                    """
            #np.save(binary_mask,)
        np.save(self.output_directory+ f"/img/mask_{img_name}" ,binary_mask)
    
        """
            elif r_id == "ipsc":
                ipsc = IdPassSegmentationColors(responds[i])
                for j in range(ipsc.get_num_segmentation_colors()):
                    print(ipsc.get_segmentation_color(j))
                    segmentation_colors_in_image.append(ipsc.get_segmentation_color(j))
        """
        for object_id in segmentation_colors_per_object:
            for i in range(len(segmentation_colors_in_image)):
                if any((segmentation_colors_in_image[i] == j).all() for j in segmentation_colors_per_object.values()):
                    #print(object_id, segmentation_color[i])
                    break

        scene_info = {}
        return scene_info
    
    def add_obj1_on_obj2(self, obj1 : str, obj2 : str):
        anchor_object_id = self.get_unique_id()
        resp = self.communicate([
                                self.get_add_object(model_name=obj1,
                                position={"x": 0, "y": 0, "z": 0.2},
                                object_id=anchor_object_id),
                                {"$type": "send_bounds",
                                            "frequency": "once"}
                            ])

        # Get the top of the table.
        top = (0, 0, 0)
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "boun":
                bounds = Bounds(resp[i])
                for j in range(bounds.get_num()):
                    if bounds.get_id(j) == anchor_object_id:
                        top = bounds.get_top(j)
                        side = bounds.get_front(j)
                        break

        company_id = self.get_unique_id()
        self.object_ids.extend([anchor_object_id, company_id])
        # Put an object on top of the table.
        return self.communicate(self.get_add_object(model_name=obj2,
                               position=TDWUtils.array_to_vector3(top),
                               object_id=company_id))
    
    def generate_vqa_pairs(self, scene_info, num_vqa_pairs = 1):
        vqa_binds = {}
        return vqa_binds


    def generate_scene_distribution(self):
        return

off_x = 0
off_y = 0
class KitchenController(Controller):    
    def __init__(self, split = "train", port = 1928, output_directory = "datasets/TDWKitchen"):
        super().__init__(port = port)
        self.moveables = []
        self.immoveables = []
        self.split = split
        self.output_directory = output_directory + f"/{split}"

        """keep track of objects generated in the scene with: id, model, texture, color etc"""
        self.object_ids = []
        self.om= ObjectManager(transforms=False, bounds=True, rigidbodies=True)
        self.rng: np.random.RandomState = np.random.RandomState(32)

        self.W, self.H = 512, 512

        split = 'train'
        self.output_directory = f"{output_directory}/{split}"

    def add_object(self, model, position = {"x": -0.1, "y": 0, "z": -0.8}, rotation = {"x": 0, "y": 0, "z": 0},scale_factor=1):
        object_id = self.get_unique_id()
        #material_record = get_material("parquet_long_horizontal_clean")
        model_record = get_model(model)
        self.communicate([
            {"$type": "add_object",
                "name": model_record.name,
                "url": model_record.get_url(),
                "scale_factor": model_record.scale_factor,
                "position": position,
                "scale_factor": scale_factor,
                "rotation": rotation,
                "category": model_record.wcategory,
                "id": object_id},
        ])
        self.object_ids.append(object_id)
    
    def generate_sequence(self, name):
        path = self.output_directory + f"/{name}"
        print(path)
        keyboard = Keyboard()
        camera = ThirdPersonCamera(position={"x": -0.5 , "y": 1.9, "z":-1.7},
                           look_at={"x": -0.3, "y": 1.3, "z": -2.2},
                           avatar_id="a")
        capture = ImageCapture(avatar_ids=["a"], path=path)

        mouse = Mouse(avatar_id="a")
        robot = Robot(name="niryo_one",
              position={"x": 0.4, "y": 1, "z": -2.5},
              rotation={"x": 0, "y": 180, "z": -1},
              robot_id= self.get_unique_id())

        self.add_ons.extend([camera, mouse, robot, keyboard, capture])

        self.communicate(self.get_add_scene(scene_name="mm_kitchen_1b"))
        
        #TODO: replace with any texture cabinet
        #self.add_object("sink_cabinet_unit_wood_oak_white_chrome_composite", position = {"x":-0.2, "y":0, "z":-2.7})        
        self.add_object("cabinet_36_two_door_wood_beech_honey_composite", position = {"x":0, "y":0, "z":-2.7})
        self.add_object("cabinet_36_two_door_wood_beech_honey_composite", position = {"x":-1.7, "y":0, "z":-2.7})
        self.add_object("carpet_rug", position = {"x":-0.2, "y":1.2, "z":-2.7},scale_factor=0.25,rotation={"x": 0, "y": 90, "z": 0})

        
        self.add_object("gas_stove", position = {"x":1.2 ,"y":0, "z":-2.8},rotation={"x": 0, "y": 90, "z": 0})
        self.add_object("b04_bowl_smooth", position = {"x":0.2 ,"y":1.3, "z":-2.7})
        self.add_object("b03_morphy_2013__vray", position = {"x":-0.6 ,"y":1.3, "z":-2.5})
        self.add_object("plate06", position = {"x":-0.6 ,"y":1.3, "z":-2.8})
        self.add_object("b04_bottle_2_max", position = {"x":-0 ,"y":1.3, "z":-3})
        
        #TODO: replace with any uniform texture/color mug
        self.add_object("coffeemug", position = {"x":0.2 ,"y":1.3, "z":-3})
        
        #TODO: replace with any uniform color pan
        self.add_object("measuring_pan", position = {"x":-0.2 ,"y":1.3, "z":-2.7})
        
        #TODO: replace with any color uniform fruit
        self.add_object("orange", position = {"x":0.3 ,"y":1.3, "z":-2.8})

        # Set the initial pose.
        global off_x, off_y
        while robot.joints_are_moving():
            self.communicate([])

        def demo():
            global off_x, off_y
            off_x = off_x +10
            print(off_x)
            robot.set_joint_targets(targets={robot.static.joint_ids_by_name["shoulder_link"]: 0 + off_x,
                                 })
        def demo2():
            global off_x, off_y
            off_x = off_x -10
            print(off_x)
            robot.set_joint_targets(targets={robot.static.joint_ids_by_name["shoulder_link"]: 0 + off_x,
                                 })
        def hitandstop1():
    
            shoulder_id = robot.static.joint_ids_by_name["shoulder_link"]
            tnow = robot.dynamic.joints[shoulder_id].angles[0]

            robot.add_joint_forces(forces={robot.static.joint_ids_by_name["shoulder_link"]: 2})


            while robot.joints_are_moving() and robot.dynamic.joints[shoulder_id].angles[0] < tnow+5 and robot.dynamic.joints[shoulder_id].angles[0] > tnow-5:
                self.communicate([])

            robot.stop_joints(joint_ids=[shoulder_id])
        def hitandstop2():
    
            shoulder_id = robot.static.joint_ids_by_name["shoulder_link"]
            tnow = robot.dynamic.joints[shoulder_id].angles[0]

            robot.add_joint_forces(forces={robot.static.joint_ids_by_name["shoulder_link"]: -2})


            while robot.joints_are_moving() and robot.dynamic.joints[shoulder_id].angles[0] < tnow+5 and robot.dynamic.joints[shoulder_id].angles[0] > tnow-5:
                self.communicate([])

            robot.stop_joints(joint_ids=[shoulder_id])
        def hit1():
            print("hit1")
            robot.add_joint_forces(forces={robot.static.joint_ids_by_name["elbow_link"]: .5,})
        def hit2():
            print("hit2")
            robot.add_joint_forces(forces={robot.static.joint_ids_by_name["elbow_link"]: -.5,})
        def arm1():
            print("hit1")
            robot.add_joint_forces(forces={robot.static.joint_ids_by_name["shoulder_link"]: .5,})
        def arm2():
            print("hit2")
            robot.add_joint_forces(forces={robot.static.joint_ids_by_name["shoulder_link"]: -.5,})

        keyboard.listen(key="Escape", commands=[{"$type": "terminate"}])
        keyboard.listen(key="W", function = demo)
        keyboard.listen(key="S", function = demo2)
        keyboard.listen(key="J", function = arm1)
        keyboard.listen(key="K", function = arm2)
        keyboard.listen(key="A", function = hit1)
        keyboard.listen(key="D", function = hit2)
        keyboard.listen(key="Q", function = hitandstop1)
        keyboard.listen(key="E", function = hitandstop2)

        done = False
        while not done:
            # End the simulation.
            if mouse.right_button_pressed:
                done = True
            # We clicked on an object.
            elif mouse.left_button_pressed and mouse.mouse_is_over_object:
                print(mouse.mouse_over_object_id)
            # Advance to the next frame.
            # Listen to the scroll wheel.
            elif mouse.scroll_wheel_delta[1] > 0:
                print("scroll up", mouse.scroll_wheel_delta)
            elif mouse.scroll_wheel_delta[0] > 0:
                print("scroll right", mouse.scroll_wheel_delta)
                off_x += mouse.scroll_wheel_delta[0]
                off_y += mouse.scroll_wheel_delta[1]
                robot.set_joint_targets(targets={robot.static.joint_ids_by_name["shoulder_link"]: 0 + off_x,
                                 robot.static.joint_ids_by_name["hand_link"]: 0 + off_y})

            self.communicate([])
        self.communicate({"$type": "terminate"})


if __name__ == "__main__":
    import csv
    import pandas as pd

    with open("metadata/object_library.csv", newline= '') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ', quotechar = '|')
        for row in reader:
            print(', '.join(row))

    df = pd.read_csv("metadata/object_library.csv")
    print(df.head(5))