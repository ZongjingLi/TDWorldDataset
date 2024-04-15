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
def transform_coordinates(item1_coord, view_coord, view_angle = None):
    #TODO: add view angle to the parameters
    transformed_item1_coord = [ view_coord[0]*item1_coord[0] - view_coord[0]*item1_coord[1], 
                                -view_coord[0]*item1_coord[0] - view_coord[1]*item1_coord[1], 
                               item1_coord[2]]   
    return transformed_item1_coord


def is_left(coord1, coord2):
    return coord1[0] < coord2[0]

def is_right(coord1, coord2):
    return coord1[0] > coord2[0]

def is_front(coord1, coord2):
    return coord1[1] < coord2[1]

def is_back(coord1, coord2):
    return coord1[1] > coord2[1]

def is_above(coord1, coord2):
    return coord1[2] > coord2[2]

def is_below(coord1, coord2):
    return coord1[2] < coord2[2]







