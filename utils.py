'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-14 10:33:17
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-15 10:08:40
 # @ Description: This file is distributed under the MIT license.
'''

"""setup the libarary of materials and models"""
from tdw.librarian import ModelLibrarian, MaterialLibrarian
full_material_library = MaterialLibrarian("materials_low.json")
full_model_library = ModelLibrarian("models_core.json")

def get_material(name):
    return full_material_library.get_record(name)

def get_model(name):
    return full_model_library.get_record(name)


"""spaital relation classifier"""
def is_left_of(pos1, pos2) -> bool:
    return True

def is_right_of(pos1, pos2) -> bool:
    return 

def is_infront_of(pos1, pos2) -> bool:
    return 

def is_behind(pos1, pos2) -> bool:
    return 

