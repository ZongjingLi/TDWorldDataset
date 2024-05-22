from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.mouse import Mouse
from tdw.add_ons.keyboard import Keyboard
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, Images,SegmentationColors, IdPassSegmentationColors
from tdw.add_ons.first_person_avatar import FirstPersonAvatar
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
"""
Click on objects to print their IDs.

"""
import time
from rinarak.utils.os import save_json, load_json
import numpy as np
import math
import torch
import matplotlib.pyplot as plt

equivalence = {
    "":["apple", "orange","banana_fix2", "jug01","b04_bowl_smooth"],
    "carpet": ["carpet_rug", "blue_rug"],
    "bigger": ["blue_satchal", "b03_basket","bakerparisfloorlamp03"],
    "plate": ["baking_sheet10",],
    "cabinet": ["cabinet_36_two_door_wood_oak_white_composite", ],
    "table": [ "b05_table_new", "small_table_green_marble",  "dining_room_table"],
    "chair": ["chair_willisau_riale", "green_side_chair", "chair_thonet_marshall"],
    "vase": ["vase_05", "kettle", "vase_01", "vase_02", "vase_03"]
}

movable_items = ["vase_05", "kettle", "apple", "orange","banana_fix2", "jug01","b04_bowl_smooth", "baking_sheet10","fork1",
                 "glass1", "knife1",   "b04_wineglass", "jug03",  "pan1","spoon1", "vase_05", "kettle", "vase_01", "vase_02", "vase_03"]

immoveable_items = [ "b05_table_new", "small_table_green_marble",  "dining_room_table", "chair_willisau_riale", "green_side_chair", "chair_thonet_marshall", "carpet_rug", "blue_rug"]


asset_map = {
    "vase_05": {"color": "white"}
}

possible_kithens = ["mm_kitchen_2b", "mm_kitchen_1b"]

random_seed = 12
# 全局变量
d = 2
angel1 = 0
angel2 = 0
output_root = "/Users/melkor/Documents/"
class MaintainController(Controller):
    def __init__(self, name="TDWHall",split = "train",room_name = "box_room_2018", load_setup = None, start_count = 0):
        super().__init__()

        self.output_root = output_root
        self.split = split
        self.name = name
        self.W, self.H = 512,512
        
        self.object_ids = []
        self.positions = []
        self.rotations = []
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

        if "Kitchen" in name:
            self.interior_scene_lighting = InteriorSceneLighting(rng=np.random.RandomState(random_seed))

            self.add_ons.extend([camera, self.mouse, self.keyboard, self.interior_scene_lighting])
        else:
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
        self.counter = start_count

        self.output_directory = self.output_root + f"datasets/{name}/{split}"

        if load_setup is not None:
            for obj_id in load_setup:
                if obj_id != "camera" and obj_id != "room_name":
                    rotate = {"x": 0, "y":0, "z":0} if "rotation" not in load_setup[obj_id] else load_setup[obj_id]["rotation"]
                    self.add_object(load_setup[obj_id]["model"], load_setup[obj_id]["position"], obj_id = int(obj_id), rotation = rotate)
                    #self.object_ids.append(obj_id)

        print("scene created, all objects loaded")
                    

        
    
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
                    """
                    for i in range(self.W):
                        for j in range(self.H):
                            #print(image[i,j] ,torch.tensor(segmentation_color),list(image[i,j]) == list(torch.tensor(segmentation_color).int()))
                            if list(image[i,j]) == list(torch.tensor(segmentation_color).int()):
                                binary_mask[i,j] = self.object_ids.index(object_id)
                    """
        np.save(self.output_directory + f"/img/mask_{img_name}" ,binary_mask)
        np.save(self.output_directory + f"/scene/ids_{img_name}", object_id_sequence)


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
        temp=self.camera_lookat
        x = d * math.sin(math.radians(angel1)) * math.cos(math.radians(angel2))
        y = d * math.sin(math.radians(angel1)) * math.sin(math.radians(angel2))
        z = d * math.cos(math.radians(angel1))
        temp = {"x": temp["x"]+0, "y": temp["y"]+0.1, "z": temp["z"]+0.}

        self.camera_lookat=temp
        self.communicate(
            {"$type": "look_at_position", "avatar_id": "a", "position": temp}
            )
        self.camera.look_at(temp)

    def decrease_d(self):
        global d,angel2,angel1
        d -= 1
        temp=self.camera_lookat
        x = d * math.sin(math.radians(angel1)) * math.cos(math.radians(angel2))
        y = d * math.sin(math.radians(angel1)) * math.sin(math.radians(angel2))
        z = d * math.cos(math.radians(angel1))
        temp = {"x": temp["x"]+0, "y": temp["y"]-0.1, "z": temp["z"]+0}

        self.camera_lookat=temp
        self.communicate(
            {"$type": "look_at_position", "avatar_id": "a", "position": temp}
            )
        self.camera.look_at(temp)

    def increase_a1(self):
        global d,angel2,angel1
        angel1 += 1
        temp=self.camera_lookat
        x = d * math.sin(math.radians(angel1)) * math.cos(math.radians(angel2))
        y = d * math.sin(math.radians(angel1)) * math.sin(math.radians(angel2))
        z = d * math.cos(math.radians(angel1))
        temp = {"x": temp["x"]+0.1, "y": temp["y"]+0., "z": temp["z"]+0.}

        self.camera_lookat=temp
        self.communicate(
            {"$type": "look_at_position", "avatar_id": "a", "position": temp}
            )
        self.camera.look_at(temp)

    def decrease_a1(self):
        global d,angel2,angel1
        angel1 -= 1
        temp=self.camera_lookat
        x = d * math.sin(math.radians(angel1)) * math.cos(math.radians(angel2))
        y = d * math.sin(math.radians(angel1)) * math.sin(math.radians(angel2))
        z = d * math.cos(math.radians(angel1))
        temp = {"x": temp["x"]-0.1, "y": temp["y"]+0, "z": temp["z"]+0.}

        self.camera_lookat=temp
        self.communicate(
            {"$type": "look_at_position", "avatar_id": "a", "position": temp}
            )
        self.camera.look_at(temp)

    def increase_a2(self):
        global d,angel2,angel1
        angel2 += 10
        temp=self.camera_lookat
        x = d * math.sin(math.radians(angel1)) * math.cos(math.radians(angel2))
        y = d * math.sin(math.radians(angel1)) * math.sin(math.radians(angel2))
        z = d * math.cos(math.radians(angel1))
        temp = {"x": temp["x"]+0., "y": temp["y"]+0., "z": temp["z"]+0.1}

        self.camera_lookat=temp
        self.communicate(
            {"$type": "look_at_position", "avatar_id": "a", "position": temp}
            )
        self.camera.look_at(temp)
  
    def decrease_a2(self):
        global d,angel2,angel1
        angel2 -= 1
        temp = self.camera_lookat
        x = d * math.sin(math.radians(angel1)) * math.cos(math.radians(angel2))
        y = d * math.sin(math.radians(angel1)) * math.sin(math.radians(angel2))
        z = d * math.cos(math.radians(angel1))
        temp = {"x": temp["x"]+0., "y": temp["y"]+0., "z": temp["z"]-0.1}
        
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
    
    def rotate_object(self):
        if self.over_object_id is not None:
            obj_idx = self.locate_object(self.over_object_id)
            self.rotations[obj_idx]["y"] += 90
            self.rotations[obj_idx]["y"] %= 360
            self.communicate(
            {"$type": "rotate_object_to_euler_angles", "id": self.over_object_id, "euler_angles": self.rotations[obj_idx]}
            )
        

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

        self.keyboard.listen(key="H", function = self.save)


        self.keyboard.listen(key="U", function = self.increase_d)
        self.keyboard.listen(key="I", function = self.decrease_d)
        self.keyboard.listen(key="K", function = self.increase_a1)
        self.keyboard.listen(key="J", function = self.decrease_a1)
        self.keyboard.listen(key="M", function = self.increase_a2)
        self.keyboard.listen(key="N", function = self.decrease_a2)

        self.keyboard.listen(key="O", function = self.replacement)
        self.keyboard.listen(key="P", function = self.generate)
        self.keyboard.listen(key="R", function = self.rotate_object)
    


    def replacement(self):
        
        names = []
        pos = []
        ids = []
        rots = []
        print(self.object_ids)
        for i in range(len(self.object_ids)):
            model_name = self.models[i]
            model_pos = self.positions[i]
            model_id = self.object_ids[i]
            model_rot = self.rotations[i]
            #print(model_name, model_id, "saved")
            if model_id is not None:

                rand_model_name = model_name
                for key in equivalence:
                    rand_class = equivalence[key]
                    if model_name in rand_class:
                        rand_model_name = np.random.choice(rand_class)
                names.append(rand_model_name)
                pos.append(model_pos)
                ids.append(model_id)
                rots.append(model_rot)

        while len(self.object_ids) > 0:
            self.delete_object(self.object_ids[0])
        for i in range(len(names)):
            #print("create:", names[i], ids[i])
            self.add_object(names[i], pos[i], ids[i], rotation = rots[i])

    def mutaion(self, pos):
        return
    
    def generate(self, num = 1000):
        self.interior_scene_lighting.reset(rng= np.random.RandomState(np.random.randint(1,5)))
        for i in range(num):
            self.replacement()
            self.save(self.output_root + f"datasets/{self.name}/{self.split}/scene/{self.counter}.json")
            self.capture()
        return 

    def reset_objects(self):self.object_ids = []

    def delete_object(self, obj_id = None):
        
        if obj_id is None: obj_id = self.over_object_id
        #print(f"try to delete object: {obj_id}")
        idx = self.locate_object(obj_id)
        self.object_ids.remove(obj_id)
        self.positions.remove(self.positions[idx])
        self.communicate([
            {"$type": "destroy_object", "id": obj_id}
        ])
        self.over_object_id = None


    def add_object(self, name, position = {"x" : 0, "y" : 0, "z" : 0}, obj_id = None, rotation = None):
        object_id = obj_id#self.get_unique_id() if obj_id is None  else obj_id
        rotation = rotation if rotation is not None else {"x":0, "y":0, "z":0} 

        self.communicate([self.get_add_object(name, object_id = object_id, position = position, rotation = rotation)])
        
        #print("add",object_id)

        self.models.append(name)
        self.positions.append(position)
        self.rotations.append(rotation)
        self.object_ids.append(object_id)

    def locate_object(self, object_id):
        assert object_id in self.object_ids
        idx = self.object_ids.index(object_id)
        #print(f"Locate Object:{idx} id:{object_id}")
        #self.communicate([
        #   {"$type": "focus_towards_object", "object_id": object_id, "speed": 0.3, "use_centroid": False, "sensor_name": "SensorContainer", "avatar_id": "a"}
        #])
        return idx

    def save(self, save_name = None):
        print("saved")
        scene_json = {}
        for i,obj_id in enumerate(self.object_ids):
            if obj_id is not None:
                scene_json[obj_id] = {
                    "model": self.models[i],
                    "position":self.positions[i],
                    "rotation": self.rotations[i],
                    "movable": self.models[i] in movable_items
                }
        scene_json["camera"] = {
            "position":self.camera_location,
            "look_at": self.camera_lookat,
            }
        scene_json["room_name"] = self.room_name
        if save_name is None:save_name = self.output_root + f"datasets/{self.name}/scene_setup.json"
        print(save_name)
        save_json(scene_json, save_name)
    
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



#mc = MaintainController(load_setup=load_json( "datasets/scene_setup_kitchen2.json"), name="TDWKitchen",start_count = 0)
#mc = MaintainController(\
#    load_setup=load_json(output_root + "datasets/TDWKitchen/scene_setup.json"),\
#    name="TDWOutroom", room_name ="iceland_beach")

mc = MaintainController(load_setup=load_json(output_root + "datasets/TDWKitchen/scene_setup.json"), name="TDWKitchen", start_count = 0)

#mc = MaintainController(load_setup=load_json(output_root + "datasets/RealisticKitchen/scene_setup.json"), name="TDWKitchen", start_count = 0)

#mc = MaintainController(load_setup=load_json("/Users/melkor/Documents/datasets/TDWHall/scene_setup.json"), name="TDWHall")
#mc = MaintainController(room_name ="tdwroom")
#mc.add_object("rh10", position={"x":0.1,"y":0.0,"z":1.0})
#mc.add_object("b04_03_077", position={"x":-0.3,"y":0.3,"z":-.4})
#mc.add_object("cgaxis_models_10_11_vray", position={"x":0.3,"y":0.9,"z":-.1})
#mc.add_object(np.random.choice(equivalence["plate"]), position={"x":0.3,"y":0.9,"z":-.1})
#mc.add_object(np.random.choice(equivalence["small_table"]), position={"x":0.0,"y":0.0,"z":-0.1})
#suburb_scene_2018 suburb_scene_2023

mc.run()
