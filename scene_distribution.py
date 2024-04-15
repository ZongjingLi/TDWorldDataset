'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-14 10:42:52
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-14 10:42:59
 # @ Description: This file is distributed under the MIT license.
'''
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Images, Transforms, Rigidbodies
from library import get_material, get_model

class SceneController(Controller):
    def __init__(self):
        super().__init__()
        self.moveables = []
        self.immoveables = []
        self.output_directory = "datasets/TDWRoom"
    
    def reset_scene(self, option = None):
        pass

    def trial(self,model_name = None, height = None, idx = 0):
        if model_name is None: model_name = "iron_box"
        if height is None: height = 0.8
        object_id = self.get_unique_id()

        commands = []
        """create an avatar to observe the environment"""
        commands.extend(TDWUtils.create_avatar(position={"x": 1.6, "y": 1.6, "z": 1.8},
                                       avatar_id="a",
                                       look_at={"x": -0.0, "y": -0.0, "z": 0.2}))
        
        """create some test objects in the scene"""
        commands.extend([
            TDWUtils.create_empty_room(5, 5),
        ])
        commands.extend([
            self.get_add_object(model_name,
                                object_id = object_id,
                                position = {"x":0, "y":height, "z":0}),
            self.get_add_object(model_name = get_model("sink_cabinet_unit_wood_beech_honey_chrome_composite").name,
                                
                                position = {'x':0.0, 'y':-0., 'z':+0.0},
                                object_id = self.get_unique_id()),
        ])
        
        
        commands.extend([
            {"$type": "set_screen_size", "width": 512, "height": 512},
            {"$type": "set_pass_masks", "pass_masks": ["_img", "_id"], "avatar_id": "a"},
            {"$type": "send_images", "frequency": "always", "ids": ["a"]}])
        
        responds = self.communicate(commands)
        for i in range(len(responds)):
            r_id = OutputData.get_data_type_id(responds[i])
            print(r_id)
            if r_id == "imag":
                image = Images(responds[i])
                avatar_id = image.get_avatar_id()
                TDWUtils.save_images(image, filename = f"scourge_{idx}", output_directory = self.output_directory)

        #self.communicate({"$type": "destroy_object", "id": object_id})
    
    def add_obj1_on_obj2(self, obj1, obj2):
        return 


    def generate_scene_distribution(self):
        return

scene_control = SceneController()

outputs = scene_control.trial()

