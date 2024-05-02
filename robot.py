'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-14 10:51:17
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-14 10:51:19
 # @ Description: This file is distributed under the MIT license.
'''

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.robot import Robot
from tdw.librarian import SceneLibrarian
import numpy as np

c = Controller()
robot_id = c.get_unique_id()
robot = Robot(name="ur5", 
              robot_id=robot_id,
              position={"x": 0., "y": 0.0, "z": -0.4},
              rotation={"x": 0, "y": 0, "z": 0},
              )
c.add_ons.append(robot)

librarian = SceneLibrarian()
for record in librarian.records:print(record.name)

c.communicate(c.get_add_scene(scene_name="tdw_room"))

commands = []
theta = np.random.random() * 2 * np.pi
scale = np.random.random() * 0.2 + 1.8
commands.extend(TDWUtils.create_avatar(position={"x": np.cos(theta) * scale, "y": 1.32, "z": np.sin(theta) * scale},
                                       avatar_id="a",
                                       look_at={"x": 0.0, "y": 0.4, "z": 0.0}))
c.communicate(commands)

