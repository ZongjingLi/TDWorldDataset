from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.mouse import Mouse
from tdw.add_ons.keyboard import Keyboard
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, Images,SegmentationColors, IdPassSegmentationColors
from tdw.add_ons.first_person_avatar import FirstPersonAvatar
"""
Click on objects to print their IDs.

"""
import time
from rinarak.utils.os import save_json, load_json
import numpy as np
import math
import torch
import matplotlib.pyplot as plt

# 全局变量
d = 2
angel1 = 0
angel2 = 0
class MaintainController(Controller):
    def __init__(self, name="TDWHall",split = "train",room_name = "box_room_2018", load_setup = None):
        super().__init__()
        self.name = name
        self.W, self.H = 512,512
        self.object_ids = []
        self.positions = []
        self.models = []

        self.room_name = room_name

        if load_setup is not None:
            self.camera_location = load_setup["camera"]["position"]
            self.camera_lookat = load_setup["camera"]["look_at"]
            self.room_name = load_setup["room_name"]
            self.communicate(self.get_add_scene(scene_name=self.room_name))
        else:
            self.camera_location = {"x": 2.478, "y": 1.602, "z": 1.412}
            self.camera_lookat = {"x": 0, "y": 0.5, "z": 0}
            self.communicate(self.get_add_scene(scene_name=self.room_name))
            #self.communicate([TDWUtils.create_empty_room(12, 12)])
        camera = ThirdPersonCamera(position=self.camera_location,
                           look_at=self.camera_lookat,
                           avatar_id="a")
        self.camera = camera
        self.mouse = Mouse(avatar_id="a")
        self.keyboard = Keyboard()
        self.add_ons.extend([camera, self.mouse, self.keyboard])
        self.setup_keyboard()
        
        self.over_object_id = None

        self.possible_object_names = [
            "small_table_green_marble",
            "rh10",
            "jug01",
            "jug05",
        ]
        self.hold_object_name = "small_table_green_marble"
        self.basic_setup()
        self.counter = 0

        self.output_directory = f"datasets/{name}/{split}"

        if load_setup is not None:
            for obj_id in load_setup:
                if obj_id != "camera" and obj_id != "room_name":
                    self.add_object(load_setup[obj_id]["model"], load_setup[obj_id]["position"])
                    self.object_ids.append(obj_id)

        
    
    def capture(self):
        
        img_name = f"{self.counter}"
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
                for j in range(segm.get_num()):
                    object_id = segm.get_object_id(j)
                    if (object_id in self.object_ids):
                        segmentation_color = segm.get_object_color(j)
                        segmentation_colors_per_object[object_id] = segmentation_color

                        locs = torch.max(id_map == torch.tensor(segmentation_color), dim = - 1, keepdim = False).values

                        binary_mask[locs] = self.object_ids.index(object_id) + 1
                    else:
                        print(object_id)
                    """
                    for i in range(self.W):
                        for j in range(self.H):
                            #print(image[i,j] ,torch.tensor(segmentation_color),list(image[i,j]) == list(torch.tensor(segmentation_color).int()))
                            if list(image[i,j]) == list(torch.tensor(segmentation_color).int()):
                                binary_mask[i,j] = self.object_ids.index(object_id)
                    """
        np.save(self.output_directory+ f"/img/mask_{img_name}" ,binary_mask)
    


        self.counter += 1
        print(f"done:{img_name}")


    def basic_setup(self):
        commands = []
        commands.extend([
    {"$type": "set_screen_size", "width":self.W, "height": self.H},
        ])
        responds = self.communicate(commands)
    

    def increase_d(self):
        global d,angel2,angel1
        d += 1
        temp=self.camera_location
        x = d * math.sin(math.radians(angel1)) * math.cos(math.radians(angel2))
        y = d * math.sin(math.radians(angel1)) * math.sin(math.radians(angel2))
        z = d * math.cos(math.radians(angel1))
        temp = {"x": temp["x"]+x, "y": temp["y"]+y, "z": temp["z"]+z}

        self.camera_lookat=temp
        self.communicate(
            {"$type": "look_at_position", "avatar_id": "a", "position": temp}
            )
        self.camera.look_at(temp)

    def decrease_d(self):
        global d,angel2,angel1
        d -= 1
        temp=self.camera_location
        x = d * math.sin(math.radians(angel1)) * math.cos(math.radians(angel2))
        y = d * math.sin(math.radians(angel1)) * math.sin(math.radians(angel2))
        z = d * math.cos(math.radians(angel1))
        temp = {"x": temp["x"]+x, "y": temp["y"]+y, "z": temp["z"]+z}

        self.camera_lookat=temp
        self.communicate(
            {"$type": "look_at_position", "avatar_id": "a", "position": temp}
            )
        self.camera.look_at(temp)

    def increase_a1(self):
        global d,angel2,angel1
        angel1 += 1
        temp=self.camera_location
        x = d * math.sin(math.radians(angel1)) * math.cos(math.radians(angel2))
        y = d * math.sin(math.radians(angel1)) * math.sin(math.radians(angel2))
        z = d * math.cos(math.radians(angel1))
        temp = {"x": temp["x"]+x, "y": temp["y"]+y, "z": temp["z"]+z}

        self.camera_lookat=temp
        self.communicate(
            {"$type": "look_at_position", "avatar_id": "a", "position": temp}
            )
        self.camera.look_at(temp)

    def decrease_a1(self):
        global d,angel2,angel1
        angel1 -= 1
        temp=self.camera_location
        x = d * math.sin(math.radians(angel1)) * math.cos(math.radians(angel2))
        y = d * math.sin(math.radians(angel1)) * math.sin(math.radians(angel2))
        z = d * math.cos(math.radians(angel1))
        temp = {"x": temp["x"]+x, "y": temp["y"]+y, "z": temp["z"]+z}

        self.camera_lookat=temp
        self.communicate(
            {"$type": "look_at_position", "avatar_id": "a", "position": temp}
            )
        self.camera.look_at(temp)

    def increase_a2(self):
        global d,angel2,angel1
        angel2 += 1
        temp=self.camera_location
        x = d * math.sin(math.radians(angel1)) * math.cos(math.radians(angel2))
        y = d * math.sin(math.radians(angel1)) * math.sin(math.radians(angel2))
        z = d * math.cos(math.radians(angel1))
        temp = {"x": temp["x"]+x, "y": temp["y"]+y, "z": temp["z"]+z}

        self.camera_lookat=temp
        self.communicate(
            {"$type": "look_at_position", "avatar_id": "a", "position": temp}
            )
        self.camera.look_at(temp)

        
    def decrease_a2(self):
        global d,angel2,angel1
        angel2 -= 1
        temp = self.camera_location
        x = d * math.sin(math.radians(angel1)) * math.cos(math.radians(angel2))
        y = d * math.sin(math.radians(angel1)) * math.sin(math.radians(angel2))
        z = d * math.cos(math.radians(angel1))
        temp = {"x": temp["x"]+x, "y": temp["y"]+y, "z": temp["z"]+z}
        
        self.camera_lookat=temp
        self.communicate(
            {"$type": "look_at_position", "avatar_id": "a", "position": temp}
            )
        self.camera.look_at(temp)


    def move_camera_up(self):
        self.camera_location["y"] += 0.1
        self.camera.teleport(position = self.camera_location)

    def move_camera_down(self):
        self.camera_location["y"] -= 0.1
        self.camera.teleport(position = self.camera_location)

    def move_object_x_plus(self):
        if self.over_object_id is not None:
            obj_idx = self.locate_object(self.over_object_id)
            self.positions[obj_idx]["x"] += 0.1
            self.communicate(
            {"$type": "teleport_object", "position": self.positions[obj_idx], "id": self.over_object_id}
            )
        else:
            self.camera_location["x"] += 0.1
            self.camera.teleport(position=self.camera_location)

    def move_object_x_minus(self):
        if self.over_object_id is not None:
            obj_idx = self.locate_object(self.over_object_id)
            self.positions[obj_idx]["x"] -= 0.1
            self.communicate(
            {"$type": "teleport_object", "position": self.positions[obj_idx], "id": self.over_object_id}
            )
        else:
            self.camera_location["x"] -= 0.1
            self.camera.teleport(position=self.camera_location)
        
    def move_object_y_plus(self):
        if self.over_object_id is not None:
            obj_idx = self.locate_object(self.over_object_id)
            self.positions[obj_idx]["y"] += 1.1
            self.communicate(
            {"$type": "teleport_object", "position": self.positions[obj_idx], "id": self.over_object_id}
            )
        else:
            self.camera_location["y"] += 0.1
            self.camera.teleport(position=self.camera_location)

    def move_object_y_minus(self):
        if self.over_object_id is not None:
            obj_idx = self.locate_object(self.over_object_id)
            self.positions[obj_idx]["y"] -= .1
            self.communicate(
            {"$type": "teleport_object", "position": self.positions[obj_idx], "id": self.over_object_id}
            )
        else:
            self.camera_location["y"] -= 0.1
            self.camera.teleport(position=self.camera_location)

    def move_object_z_plus(self):
        if self.over_object_id is not None:
            obj_idx = self.locate_object(self.over_object_id)
            self.positions[obj_idx]["z"] += .1
            self.communicate(
            {"$type": "teleport_object", "position": self.positions[obj_idx], "id": self.over_object_id}
            )
        else:
            self.camera_location["z"] += 0.1
            self.camera.teleport(position=self.camera_location)

    def move_object_z_minus(self):
        if self.over_object_id is not None:
            obj_idx = self.locate_object(self.over_object_id)
            self.positions[obj_idx]["z"] -= .1
            self.communicate(
            {"$type": "teleport_object", "position": self.positions[obj_idx], "id": self.over_object_id}
            )
        else:
            self.camera_location["z"] -= 0.1
            self.camera.teleport(position=self.camera_location)
    
    def reset_cursor(self): self.over_object_id = None

    def setup_keyboard(self):
        self.keyboard.listen(key="Escape", commands = [{"$type": "terminate"}])
        self.keyboard.listen(key="W", function = self.move_object_x_plus)
        self.keyboard.listen(key="Z", function = self.move_object_x_minus)

        self.keyboard.listen(key="E", function = self.move_object_y_plus)
        self.keyboard.listen(key="D", function = self.move_object_y_minus)

        self.keyboard.listen(key="S", function = self.move_object_z_plus)
        self.keyboard.listen(key="A", function = self.move_object_z_minus)

        self.keyboard.listen(key="C", function = self.capture)

        self.keyboard.listen(key="T", function = self.delete_object)
        self.keyboard.listen(key="G", function = self.create_object)

        self.keyboard.listen(key="L", function = self.save)


        self.keyboard.listen(key="I", function = self.increase_d)
        self.keyboard.listen(key="O", function = self.decrease_d)
        self.keyboard.listen(key="K", function = self.increase_a1)
        self.keyboard.listen(key="J", function = self.decrease_a1)
        self.keyboard.listen(key="M", function = self.increase_a2)
        self.keyboard.listen(key="N", function = self.decrease_a2)


    def reset_objects(self):self.object_ids = []

    def delete_object(self):
        print(f"try to delete object: {self.over_object_id}")
        idx = self.locate_object(self.over_object_id)
        self.object_ids[idx] = None
        self.positions[idx] = None
        self.communicate([
            {"$type": "destroy_object", "id": self.over_object_id}
        ])
        self.over_object_id = None

    def create_object(self):
        #self.hold_object_name
        #object_name = self.hold_object_name
        #pos = {"x":0.0,"y":0.0,"z":0.0}
        #print(object_name)
        #time.sleep(0.1)
        #return self.add_object(object_name, position = pos)
        pass    

    def add_object(self, name, position = {"x" : 0, "y" : 0, "z" : 0}):
        object_id = self.get_unique_id()
        self.communicate([self.get_add_object(name, object_id = object_id, position = position)])
        
        self.models.append(name)
        self.positions.append(position)
        self.object_ids.append(object_id)

    def locate_object(self, object_id):
        assert object_id in self.object_ids
        idx = self.object_ids.index(object_id)
        #print(f"Locate Object:{idx} id:{object_id}")
        #self.communicate([
        #   {"$type": "focus_towards_object", "object_id": object_id, "speed": 0.3, "use_centroid": False, "sensor_name": "SensorContainer", "avatar_id": "a"}
        #])
        return idx

    def save(self):
        scene_json = {}
        for i,obj_id in enumerate(self.object_ids):
            if obj_id is not None:
                scene_json[obj_id] = {
                    "model": self.models[i],
                    "position":self.positions[i],
                }
        scene_json["camera"] = {
            "position":self.camera_location,
            "look_at": self.camera_lookat,
            }
        scene_json["room_name"] = self.room_name
        print(scene_json)
        save_json(scene_json,f"datasets/{self.name}/scene_setup.json")
    
    def run(self):
        done = False
        mouse = self.mouse
        while not done:
            # End the simulation.
            if mouse.right_button_pressed:
                self.reset_cursor()
            # We clicked on an object.
            elif mouse.left_button_pressed and mouse.mouse_is_over_object:
                print(mouse.mouse_over_object_id)
                self.over_object_id = mouse.mouse_over_object_id
                self.locate_object(self.over_object_id)
            # Advance to the next frame.
            self.communicate([])
        self.communicate({"$type": "terminate"})
    
    def replace_with_equivalence(self, equivalence_table):
        return 

equivalence = {
    "plate": ["round_bowl_small_beech", "plate05"],
    "small_table": ["dining_room_table", "small_table_green_marble","lg_table_white"],
    "painting":["elf_painting","framed_painting","its_about_time_painting"],
    "round_table":["enzo_industrial_loft_pine_metal_round_dining_table"],
    "rug": ["flat_woven_rug", "purple_woven_rug"],
    "bench":["glass_table", "quatre_dining_table"]
}

mc = MaintainController(load_setup=load_json("datasets/TDWHall/scene_setup.json"))
#mc = MaintainController()
#mc.add_object("rh10", position={"x":0.1,"y":0.0,"z":1.0})
#mc.add_object("jug01", position={"x":0.1,"y":0.9,"z":-.1})
#mc.add_object("apple", position={"x":0.3,"y":0.9,"z":-.1})
#mc.add_object(np.random.choice(equivalence["plate"]), position={"x":0.3,"y":0.9,"z":-.1})
#mc.add_object(np.random.choice(equivalence["small_table"]), position={"x":0.0,"y":0.0,"z":-0.1})
#mc.add_object("small_table_green_marble", position={"x":1.5,"y":0.0,"z":-.4})


mc.run()
