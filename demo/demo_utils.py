import os
from unicodedata import name

class TrainedModel(object):
    def __init__(self, name, modeltypes):
        self.name = name 
        self.modeltypes = modeltypes
        for i in range(len(self.modeltypes)):
            self.modeltypes[i] = self.modeltypes[i].capitalize()
            

    def __repr__(self) -> str:
        return f"{self.name}: {self.modeltypes}" 


def findModel(name, model_objects):
    for model in model_objects:
        if name == model.name.lower():
            return model
    
    return None


def getModelNames(model_objects):
    names = []
    for model in model_objects:
        names.append(model.name.capitalize())
    return names



def createModels():
    model_path = os.path.join(os.getcwd(), "demo", "trained_models")
    
    files = os.listdir(model_path)


    models = []
    model_to_types = dict()
    for file in files:
        curr_path = os.path.join(model_path, file)
        if os.path.isfile(curr_path):
            if file.endswith(".pt"):
                name_type = file.split("_")
                model_name = name_type[0]
                type = name_type[1][:name_type[1].find(".pt")]
                if model_name not in models:
                    models.append(model_name)
                if model_name in model_to_types:
                    curr_list = model_to_types[model_name]
                    curr_list.append(type)
                    model_to_types[model_name] = curr_list
                else: 
                    model_to_types[model_name] = [type]
    
    list_TrainedModels = []
    for model in models: 
        list_TrainedModels.append(TrainedModel(model, model_to_types[model]))
    return list_TrainedModels