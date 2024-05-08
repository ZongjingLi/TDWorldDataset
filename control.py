from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.robot import Robot
from tdw.add_ons.mouse import Mouse
from tdw.add_ons.keyboard import Keyboard
from tdw.add_ons.third_person_camera import ThirdPersonCamera
import numpy as np
from utils import get_material, get_model

def add_object(c, model, position = {"x": -0.1, "y": 0, "z": -0.8}):
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
                "rotation": {"x": 0, "y": 0, "z": 0},
                "category": model_record.wcategory,
                "id": object_id},
            ])

c = Controller()
keyboard = Keyboard()
camera = ThirdPersonCamera(position={"x": 1.0 , "y": 2.0, "z":3.2},
                           look_at={"x": 1.0, "y": 1.3, "z": 2},
                           avatar_id="a")

mouse = Mouse(avatar_id="a")
robot = Robot(name="niryo_one",
              position={"x": 1.6, "y": 1, "z": 2.1},
              rotation={"x": 0, "y": 0, "z": 0},
              robot_id=c.get_unique_id())

c.add_ons.extend([camera, mouse, robot, keyboard])


#c.communicate(TDWUtils.create_empty_room(12, 12))

c.communicate(c.get_add_scene(scene_name="mm_kitchen_1b"))
add_object(c, "sink_cabinet_unit_wood_oak_white_chrome_composite", position = {"x":1, "y":0, "z":2})
add_object(c, "gas_stove", position = {"x":2 ,"y":0, "z":-1})



for joint_id in robot.static.joints:
    joint_name = robot.static.joints[joint_id].name
    joint_mass = robot.static.joints[joint_id].mass
    joint_segmentation_color = robot.static.joints[joint_id].segmentation_color
    joint_type = robot.static.joints[joint_id].joint_type
    print(joint_name, joint_mass, joint_segmentation_color, joint_type)

# Set the initial pose.
while robot.joints_are_moving():
    c.communicate([])

# Strike a cool pose.
#robot.set_joint_targets(targets={robot.static.joint_ids_by_name["shoulder_link"]: 50,
#                                 robot.static.joint_ids_by_name["forearm_link"]: -60})



# Wait for the joints to stop moving.
while robot.joints_are_moving():
    c.communicate([])

off_x = 0
off_y = 0



def demo():
    print("wow")
    #robot.static.joint_ids_by_name["hand_link"]

    robot.set_joint_targets(targets={robot.static.joint_ids_by_name["shoulder_link"]: 130 + off_x,
                                 robot.static.joint_ids_by_name["hand_link"]: 160 + off_y})

keyboard.listen(key="Escape", commands=[{"$type": "terminate"}])
keyboard.listen(key="W", function = demo)
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