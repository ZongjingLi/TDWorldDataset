'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-14 10:42:52
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-14 10:42:59
 # @ Description: This file is distributed under the MIT license.
'''
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Images, Transforms, Rigidbodies, Bounds
from tdw.add_ons.object_manager import ObjectManager
from utils import get_material, get_model

import math
import random
import numpy as np

from typing import List

class SceneController(Controller):
    def __init__(self, split = "train", port = 1027, output_directory = "datasets/TDWRoom"):
        super().__init__(port = port)
        self.moveables = []
        self.immoveables = []
        self.split = split
        self.output_directory = output_directory + f"/{split}"

        """keep track of objects generated in the scene with: id, model, texture, color etc"""
        self.object_ids = []
        self.om= ObjectManager(transforms=False, bounds=True, rigidbodies=True)
        self.rng: np.random.RandomState = np.random.RandomState(32)

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

    def generate_scene(self,model_name = None, height = None, img_name = 0):
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
        commands.extend([
            TDWUtils.create_empty_room(12, 12),
        ])
        self.communicate(self.set_floor())
        self.communicate(self.set_walls())

        """add some objects in the scene"""
        object_id = self.get_unique_id()
        self.object_ids.append(object_id)
        commands.extend([
            self.get_add_object(model_name,
                                object_id = object_id,
                                position = {"x":0, "y":height, "z":0}),
        ])

        self.add_obj1_on_obj2("small_table_green_marble", "jug01")


        object_id = self.get_unique_id()
        material_record = get_material("parquet_long_horizontal_clean")
        model_record = get_model("white_lounger_chair")
        self.communicate([
            {"$type": "add_object",
                "name": model_record.name,
                "url": model_record.get_url(),
                "scale_factor": model_record.scale_factor,
                "position": {"x": -0.1, "y": 0, "z": -0.8},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "category": model_record.wcategory,
                "id": object_id},
        ])
        self.communicate([
            {"$type": "set_visual_material",
                "material_index": 0,
                "material_name": material_record.name,
                "object_name": "Object017",
                "id": object_id}
        ])
        self.object_ids.append(object_id)
    
        
        commands.extend([
            {"$type": "set_screen_size", "width": 512, "height": 512},
            {"$type": "set_pass_masks", "pass_masks": ["_img", "_id"], "avatar_id": "a"},
            {"$type": "send_images", "frequency": "always", "ids": ["a"]}])
        
        responds = self.communicate(commands)
        for i in range(len(responds)):
            r_id = OutputData.get_data_type_id(responds[i])
            if r_id == "imag":
                image = Images(responds[i])
                avatar_id = image.get_avatar_id()
                TDWUtils.save_images(image, filename = f"{img_name}", output_directory = self.output_directory+ f"/img")
        
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

