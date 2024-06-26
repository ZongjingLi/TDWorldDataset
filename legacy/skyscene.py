from tdw.controller import Controller
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture

c = Controller()
camera = ThirdPersonCamera(avatar_id="a",
                           position={"x": -4.28, "y": 0.85, "z": 4.27},
                           look_at={"x": 0, "y": 0, "z": 0})
path = "datasets/hdri_skybox"
print(f"Image will be saved to: {path}")
capture = ImageCapture(avatar_ids=["a"], path=path)
c.add_ons.extend([camera, capture])
c.communicate([{"$type": "set_screen_size",
                "width": 512,
                "height": 512},
               c.get_add_scene(scene_name="building_site"),
               c.get_add_hdri_skybox(skybox_name="bergen_4k")])
for hdri_skybox in ["industrial_sunset_4k", "misty_pines_4k", "harties_4k"]:
    c.communicate(c.get_add_hdri_skybox(hdri_skybox))
c.communicate({"$type": "terminate"})
"""
one can adopt microservices architecture. it means that we can break the functionalities into components that 
works interact with each other using APIs. These individual components are independent of each other and that 
makes the app scalable and flexibly when setting on different platforms.
"""