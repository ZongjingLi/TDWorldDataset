from time import sleep
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Images
from pathlib import Path
from tdw.output_data import OutputData, SceneRegions
import datetime
from tdw.librarian import ModelLibrarian, MaterialLibrarian
import random
import math
import json
material_record = MaterialLibrarian("materials_low.json").get_record("parquet_long_horizontal_clean")
full_library = ModelLibrarian("models_core.json")
c = Controller()



#三个的表格
materials = ["bronze_yellow", "wood_teak", "plastic_vinyl_glossy_yellow"]
colors = ["red", "green", "blue"]
categories = ["iron_box", "alarm_clock", "b03_aluminum_pan"]


color_mapping = {
    "red": {"r": 1.0, "g": 0, "b": 0, "a": 1.0},
    "green": {"r": 0, "g": 1, "b": 0, "a": 1.0},
    "blue": {"r": 0, "g": 0, "b": 1, "a": 1.0}
}

result = {}
#设定所有物品三个
for i in range(1, 6):
    choices = {}
    choices["material"] = random.choice(materials)
    choices["color"] = random.choice(colors)
    choices["category"] = random.choice(categories)
    result[i] = choices
# Generate a unique object ID.
object_ids = [
    c.get_unique_id(),
    c.get_unique_id(),
    c.get_unique_id(),
    c.get_unique_id(),
    c.get_unique_id()
]


commands = [TDWUtils.create_empty_room(12, 12),
                
              

                  
               
               ]
#设置创建物品并设置颜色
for i in range(1, 6):
    commands.extend([c.get_add_object(model_name=result[i]["category"],
                                library="models_core.json",
                                position={"x": random.uniform(-1.5, 1.5), "y": random.uniform(-1.5, 1.5), "z": random.uniform(0, 1)},
                                object_id=object_ids[i-1]),])



    model_record = ModelLibrarian().get_record(result[i]["category"])

#设置材质，但是substructure
    commands.extend(
    #TDWUtils.set_visual_material(c=c, substructure=model_record.substructure, material="stone_surface_rough_cracks", object_id=object_id),
        TDWUtils.set_visual_material(c=c, substructure=model_record.substructure, material=result[i]["material"], object_id=object_ids[i-1]),

        )
#根据颜色改变颜色
    color = result[i]["color"]

    if color in color_mapping:
        commands.extend([{
            "$type": "set_color",
            "color": color_mapping[color],
            "id": object_ids[i-1]
        }])



#摄像头和输出
commands.extend(TDWUtils.create_avatar(position={"x": 3, "y":3, "z": 3},
                                       avatar_id="a",
                                       look_at={"x": 0.0, "y": 0.0, "z": 0}))
commands.extend([{"$type": "set_pass_masks",
                  "pass_masks": ["_category", "_id","_img"],
                  #_img
                  "avatar_id": "a"},
                 {"$type": "send_images",
                  "frequency": "always",
                  "ids": ["a"]}])

commands.extend([{"$type": "set_screen_size",
                "width": 512,
                "height": 512}])
# 通过与TDW控制器通信发送命令并获取响应
resp = c.communicate(commands)

# 设置输出目录
output_directory = str(Path.cwd().joinpath("datasets/ExampleImages"))
print(f"Images will be saved to: {output_directory}")

# 生成时间戳命名文件
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filenameb = f"file_{timestamp}"

# 遍历响应数据
for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])

    # 提取图像数据
    if r_id == "imag":
        images = Images(resp[i])

        # 判断图像的来源
        if images.get_avatar_id() == "a":
            # 保存图像
            TDWUtils.save_images(images=images, filename=filenameb, output_directory=output_directory)



c.communicate({"$type": "terminate"})
json_data = json.dumps(result, indent=4)
file_path = f"C:/Users/DELL/datasets/ExampleImages/{filenameb}.json"  # 替换为您想要保存的文件路径
with open(file_path, "w") as file:
    file.write(json_data)