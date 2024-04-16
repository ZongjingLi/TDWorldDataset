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
#第二项代表高度，无需变换，通过变化第一项和第三项使得摄像头从（1，0，0）看向（0，0，0）
 #假如摄像头在（a,h,c），一个在（a,0,c）坐标的物品显然在原点的正"前"方
 #（a，0，c）————>(1,0,0)
 #(-c,0,a)在正右方，应当变为（0，0，1）
 #所以变换后的坐标应当为（ax/(aa+cc)+cy/(aa+cc),0，-cx/(aa+cc)+ay/(aa+cc)）
    #view_coord[0]*item1_coord[0] - view_coord[0]*item1_coord[1]
    transformed_item1_coord = [ view_coord[0]*item1_coord[0]/(view_coord[0]^2+view_coord[2]^2)+view_coord[2]*item1_coord[2]/(view_coord[0]^2+view_coord[2]^2), 
                                item1_coord[1], 
                               -view_coord[2]*item1_coord[0]/(view_coord[0]^2+view_coord[2]^2)+view_coord[0]*item1_coord[2]/(view_coord[0]^2+view_coord[2]^2)]   
    return transformed_item1_coord




#当camera从（1，0，0）看向（0，0，0）时
#（0，0，-1）是左
def is_left(coord1, coord2):
    return coord1[2] < coord2[2]

def is_right(coord1, coord2):
    return coord1[2] > coord2[2]
#(1,0,0)是前
def is_front(coord1, coord2):
    return coord1[0] > coord2[0]

def is_back(coord1, coord2):
    return coord1[0] < coord2[0]
#（0，1，0）是上
def is_above(coord1, coord2):
    return coord1[1] > coord2[1]

def is_below(coord1, coord2):
    return coord1[1] < coord2[1]







