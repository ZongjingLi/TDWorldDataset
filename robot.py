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

c = Controller()
robot_id = c.get_unique_id()
robot = Robot(name="ur5", robot_id=robot_id)
c.add_ons.append(robot)
commands = [TDWUtils.create_empty_room(12, 12)]
commands.extend(TDWUtils.create_avatar(avatar_id="avatar"))
c.communicate(commands)