import numpy as np
from tdw.controller import Controller
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.add_ons.mouse import Mouse
from tdw.add_ons.keyboard import Keyboard

path = "datasets/RealisticKitchen"
proc_gen_kitchen = ProcGenKitchen()
random_seed = 12
proc_gen_kitchen.create(rng=np.random.RandomState(random_seed))
print(proc_gen_kitchen.commands)
interior_scene_lighting = InteriorSceneLighting(rng=np.random.RandomState(random_seed))


loc = {"x": -1, "y": 1.8, "z": 2}

camera = ThirdPersonCamera(position=loc,
                           look_at={"x": 0, "y": 1, "z": 0},
                           avatar_id="a")
#capture = ImageCapture(avatar_ids=["a"], path=path, pass_masks=["_img"])
mouse = Mouse(avatar_id="a")
keyboard = Keyboard()

c = Controller()
c.add_ons.extend([proc_gen_kitchen, interior_scene_lighting, camera,  mouse, keyboard])

def w_func():
    loc["x"] += 0.1
    camera.teleport(loc)
def z_func():
    loc["x"] -= 0.1
    camera.teleport(loc)
def a_func():
    loc["z"] += 0.1
    camera.teleport(loc)
def s_func():
    loc["z"] -= 0.1
    camera.teleport(loc)
def e_func():
    loc["y"] += 0.1
    camera.teleport(loc)
def d_func():
    loc["y"] -= 0.1
    camera.teleport(loc)

keyboard.listen("W", w_func)
keyboard.listen("Z", z_func)
keyboard.listen("A", a_func)
keyboard.listen("S", s_func)
keyboard.listen("E", e_func)
keyboard.listen("D", d_func)

c.communicate([{"$type": "set_screen_size",
                "width": 512,
                "height": 512}])
done = False

while not done:
    # End the simulation.
    if mouse.right_button_pressed:
        #c.reset_cursor()
        pass
    # We clicked on an object.
    elif mouse.left_button_pressed and mouse.mouse_is_over_object:
        print(mouse.mouse_over_object_id)
        responds = c.communicate()
       # c.over_object_id = mouse.mouse_over_object_id
        #c.locate_object(c.over_object_id)
    c.communicate([])
c.communicate({"$type": "terminate"})