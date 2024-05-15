import numpy as np
from tdw.controller import Controller
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH

from tdw.add_ons.keyboard import Keyboard
from tdw.add_ons.mouse import Mouse

path ="datasets/proc_gen_kitchen_lighting"
print(f"Images will be saved to: {path}")

W, H = (512, 512)
camera_location = {"x": -2, "y": 1.8, "z": -0.5}

flag = True
if flag:
    proc_gen_kitchen = ProcGenKitchen()
    proc_gen_kitchen.create(scene="mm_kitchen_2b",rng=0)
    camera = ThirdPersonCamera(position=camera_location,
                           look_at={"x": 0, "y": 0.6, "z": 0},
                           avatar_id="a")
    capture = ImageCapture(avatar_ids=["a"], path=path, pass_masks=["_img"])
    c = Controller()
    keyboard = Keyboard()
    mouse = Mouse(avatar_id="a")
    c.add_ons.extend([proc_gen_kitchen, camera, capture, mouse, keyboard])

    """setup the keyboard"""
    camera_location["y"] += 0.1
    camera.teleport(position=camera_location)

    commands = []
    commands.extend([
    {"$type": "set_screen_size", "width": W, "height": H},
    ])
    c.communicate(commands)

    #c.communicate({"$type": "terminate"})
    done = False
    while not done:
        # End the simulation.
        if mouse.right_button_pressed:
                c.reset_cursor()
            # We clicked on an object.
        elif mouse.left_button_pressed and mouse.mouse_is_over_object:
            pass
            # Advance to the next frame.
        c.communicate([])
    c.communicate({"$type": "terminate"})
else:
    proc_gen_kitchen = ProcGenKitchen()
    random_seed = 14
    proc_gen_kitchen.create(rng=np.random.RandomState(random_seed))
    interior_scene_lighting = InteriorSceneLighting(rng=np.random.RandomState(random_seed))
    camera = ThirdPersonCamera(position={"x": -1, "y": 1.8, "z": 2},
                           look_at={"x": 0, "y": 1, "z": 0},
                           avatar_id="a")
    capture = ImageCapture(avatar_ids=["a"], path=path, pass_masks=["_img"])
    c = Controller()
    c.add_ons.extend([proc_gen_kitchen, interior_scene_lighting, camera, capture])
    c.communicate([{"$type": "set_screen_size",
                "width": 720,
                "height": 720}])
    c.communicate({"$type": "terminate"})