'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-16 07:39:02
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-16 07:39:12
 # @ Description: This file is distributed under the MIT license.
'''

from tdw.librarian import ModelLibrarian
from utils import get_material, get_model


record = get_model("034_vray")
record = get_model("small_table_green_marble")
for structure in record.substructure:print(structure)