from tdw.controller import Controller
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.output_data import OutputData, Images
from tdw.tdw_utils import TDWUtils

from library import get_material, get_model

output_directory = "datasets/TDWRoom"
commands = []

camera = ThirdPersonCamera(position={"x": 2, "y": 1.8, "z": -0.5},
                           look_at={"x": 0, "y": 0.6, "z": 0},
                           avatar_id="a")

c = Controller()

name = "sink_cabinet_unit_wood_beech_honey_chrome_composite"

commands.extend([
    TDWUtils.create_empty_room(12, 12),

    c.get_add_object(model_name = get_model(name).name,
                            position = {'x':0.5, 'y':-0., 'z':+0.4},
                            object_id = c.get_unique_id()),
])


commands.extend(TDWUtils.create_avatar(position={"x": 1.8, "y": 1.6, "z": 1.8},
                                       avatar_id="a",
                                       look_at={"x": -0.0, "y": -0.0, "z": 0.2}))

commands.extend([
    {"$type": "set_screen_size", "width": 512, "height": 512},
    {"$type": "set_pass_masks", "pass_masks": ["_img", "_id"], "avatar_id": "a"},
    {"$type": "send_images", "frequency": "always", "ids": ["a"]}])

resp = c.communicate(commands)

for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    # Get Images output data.
    if r_id == "imag":
        images = Images(resp[i])
        # Determine which avatar captured the image.
        if images.get_avatar_id() == "a":
            TDWUtils.save_images(images=images, filename="3", output_directory=output_directory)
c.communicate({"$type": "terminate"})