from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.robot import Robot
from tdw.add_ons.mouse import Mouse
from tdw.add_ons.keyboard import Keyboard
from tdw.add_ons.third_person_camera import ThirdPersonCamera
import numpy as np
from utils import get_material, get_model
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
from tdw.add_ons.image_capture import ImageCapture
from tdw.output_data import OutputData, Images, Transforms, Rigidbodies, Bounds, SegmentationColors, IdPassSegmentationColors

import torch
import matplotlib.pyplot as plt
split = 'train'
output_directory = f"datasets/TDWKitchen/{split}"

def add_object(c, model, position = {"x": -0.1, "y": 0, "z": -0.8}, rotation = {"x": 0, "y": 0, "z": 0}):
        object_id = c.get_unique_id()
        material_record = get_material("parquet_long_horizontal_clean")
        model_record = get_model(model)
        if model_record is not None:
            c.communicate([
            {"$type": "add_object",
                "name": model_record.name,
                "url": model_record.get_url(),
                "scale_factor": model_record.scale_factor,
                "position": position,
                "rotation": rotation,
                "category": model_record.wcategory,
                "id": object_id},
            ])
img_name = 0
c = Controller()
keyboard = Keyboard()
camera = ThirdPersonCamera(position={"x": -0.5 , "y": 1.9, "z":-1.7},
                           look_at={"x": -0.3, "y": 1.3, "z": -2.2},
                           avatar_id="a")
path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("robot_add_on")
print(f"Images will be saved to: {path}")
capture = ImageCapture(avatar_ids=["a"], path=path)



mouse = Mouse(avatar_id="a")
robot = Robot(name="niryo_one",
              position={"x": 0.4, "y": 1, "z": -2.5},
              rotation={"x": 0, "y": 180, "z": -1},
              robot_id=c.get_unique_id())

c.add_ons.extend([camera, mouse, robot, keyboard, capture])


#c.communicate(TDWUtils.create_empty_room(12, 12))

c.communicate(c.get_add_scene(scene_name="mm_kitchen_1b"))
#add_object(c, "sink_cabinet_unit_wood_oak_white_chrome_composite", position = {"x":-0.2, "y":0, "z":-2.7})

add_object(c, "cabinet_36_two_door_wood_beech_honey_composite", position = {"x":-0.2, "y":0, "z":-2.7})
#  cabinet_36_two_door_wood_beech_honey_composite
add_object(c, "gas_stove", position = {"x":1.2 ,"y":0, "z":-2.8},rotation={"x": 0, "y": 90, "z": 0})
add_object(c, "b04_bowl_smooth", position = {"x":0.2 ,"y":1.2, "z":-2.7})
add_object(c, "b03_morphy_2013__vray", position = {"x":-0.6 ,"y":1.2, "z":-2.5})
add_object(c, "plate06", position = {"x":-0.6 ,"y":1.2, "z":-2.8})
add_object(c, "b04_bottle_2_max", position = {"x":-0 ,"y":1.2, "z":-3})
add_object(c, "coffeemug", position = {"x":0.2 ,"y":1.2, "z":-3})
add_object(c, "measuring_pan", position = {"x":-0.2 ,"y":1.2, "z":-2.7})
add_object(c, "orange", position = {"x":0.3 ,"y":1.2, "z":-2.8})
#add_object(c, "glass_table", position = {"x":0 ,"y":0, "z":-1.8})
#add_object(c, "b04_bowl_smooth", position = {"x":1 ,"y":2, "z":0})

for joint_id in robot.static.joints:
    joint_name = robot.static.joints[joint_id].name
    joint_mass = robot.static.joints[joint_id].mass
    joint_segmentation_color = robot.static.joints[joint_id].segmentation_color
    joint_type = robot.static.joints[joint_id].joint_type
    print(joint_name, joint_mass, joint_segmentation_color, joint_type)

W=512
H=512
commands = []
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

responds = c.communicate(commands)
segmentation_colors_per_object = dict()
segmentation_colors_in_image = list()

binary_mask = torch.zeros([W, H])

for i in range(len(responds)):
    r_id = OutputData.get_data_type_id(responds[i])
    if r_id == "imag":
        image = Images(responds[i])
        avatar_id = image.get_avatar_id()
        TDWUtils.save_images(image, filename = f"{img_name}", output_directory = output_directory+ f"/img")

image = (torch.tensor(plt.imread(output_directory + f"/img/id_{img_name}.png")) * 255).int()

for i in range(len(responds)):
    r_id = OutputData.get_data_type_id(responds[i])
    if r_id == "segm":
        segm = SegmentationColors(responds[i])
        for j in range(segm.get_num()):
            object_id = segm.get_object_id(j)
            #object_name = self.object_ids.index(object_id)
            segmentation_color = segm.get_object_color(j)
            segmentation_colors_per_object[object_id] = segmentation_color
            #print(self.object_ids.index(object_id), segmentation_color)
            #print((torch.tensor(image[:,:]) == segmentation_color).shape)
            locs = torch.max(image == torch.tensor(segmentation_color), dim = - 1, keepdim = False).values
            #print(locs.shape)
            #print(binary_mask.shape)
            binary_mask[locs] = c.object_ids.index(object_id) + 1
            """
            for i in range(self.W):
                for j in range(self.H):
                    #print(image[i,j] ,torch.tensor(segmentation_color),list(image[i,j]) == list(torch.tensor(segmentation_color).int()))
                    if list(image[i,j]) == list(torch.tensor(segmentation_color).int()):
                        binary_mask[i,j] = self.object_ids.index(object_id)
            """
    #np.save(binary_mask,)
np.save(c.output_directory+ f"/img/mask_{img_name}" ,binary_mask)

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











# Set the initial pose.
while robot.joints_are_moving():
    c.communicate([])

# Strike a cool pose.
#robot.set_joint_targets(targets={robot.static.joint_ids_by_name["shoulder_link"]: 50,
#                                 robot.static.joint_ids_by_name["forearm_link"]: -60})


off_x = 0
off_y = 0



def demo():

    global off_x, off_y
    off_x = off_x +10
    print(off_x)
    robot.set_joint_targets(targets={robot.static.joint_ids_by_name["shoulder_link"]: 0 + off_x,
                                 })
def demo2():

    global off_x, off_y
    #robot.static.joint_ids_by_name["hand_link"]
    off_x = off_x -10
    #off_y = off_y - 10
    print(off_x)
    robot.set_joint_targets(targets={robot.static.joint_ids_by_name["shoulder_link"]: 0 + off_x,
                                 #robot.static.joint_ids_by_name["hand_link"]: 0 + off_y
                                 })
    
def hitandstop1():
    
    shoulder_id = robot.static.joint_ids_by_name["shoulder_link"]
    tnow = robot.dynamic.joints[shoulder_id].angles[0]

    robot.add_joint_forces(forces={robot.static.joint_ids_by_name["shoulder_link"]: 2})


    while robot.joints_are_moving() and robot.dynamic.joints[shoulder_id].angles[0] < tnow+5 and robot.dynamic.joints[shoulder_id].angles[0] > tnow-5:
        c.communicate([])

    robot.stop_joints(joint_ids=[shoulder_id])
    
def hitandstop2():
    
    shoulder_id = robot.static.joint_ids_by_name["shoulder_link"]
    tnow = robot.dynamic.joints[shoulder_id].angles[0]

    robot.add_joint_forces(forces={robot.static.joint_ids_by_name["shoulder_link"]: -2})


    while robot.joints_are_moving() and robot.dynamic.joints[shoulder_id].angles[0] < tnow+5 and robot.dynamic.joints[shoulder_id].angles[0] > tnow-5:
        c.communicate([])

    robot.stop_joints(joint_ids=[shoulder_id])
    



'''
def arm1():
    print("wow")
    global off_x, off_y
    #robot.static.joint_ids_by_name["hand_link"]
    off_y = off_y -10
    #off_y = off_y - 10
    print(off_x)
    robot.set_joint_targets(targets={robot.static.joint_ids_by_name["arm_link"]: 0 + off_y,
                                 #robot.static.joint_ids_by_name["hand_link"]: 0 + off_y
                                 })
def arm2():
    print("wow")
    global off_x, off_y
    #robot.static.joint_ids_by_name["hand_link"]
    off_y = off_y +10
    #off_y = off_y - 10
    print(off_x)
    robot.set_joint_targets(targets={robot.static.joint_ids_by_name["arm_link"]: 0 + off_y,
                                 #robot.static.joint_ids_by_name["hand_link"]: 0 + off_y
                                 })
'''


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



                               


def turnleft():
    print("tryleft")



keyboard.listen(key="Escape", commands=[{"$type": "terminate"}])
keyboard.listen(key="W", function = demo)
keyboard.listen(key="S", function = demo2)
keyboard.listen(key="J", function = arm1)
keyboard.listen(key="K", function = arm2)
keyboard.listen(key="A", function = hit1)
keyboard.listen(key="D", function = hit2)
keyboard.listen(key="Q", function = hitandstop1)
keyboard.listen(key="E", function = hitandstop2)
#c.communicate({"$type": "terminate"})

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

    c.communicate([])
c.communicate({"$type": "terminate"})


#python f:\cofemaking\TDWorldDataset\control.py