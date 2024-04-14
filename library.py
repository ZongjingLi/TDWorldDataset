"""setup the libarary of materials and models"""
from tdw.librarian import ModelLibrarian, MaterialLibrarian
full_material_library = MaterialLibrarian("materials_low.json")
full_model_library = ModelLibrarian("models_core.json")

def get_material(name):
    return full_material_library.get_record(name)

def get_model(name):
    return full_model_library.get_record(name)