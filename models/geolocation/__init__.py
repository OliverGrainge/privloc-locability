from models.geolocation.geoclip import GeoCLIPModel

def load_model(model_name: str): 
    model_name = model_name.lower()
    if model_name == "geoclip":
        return GeoCLIPModel()
    else:
        raise ValueError(f"Model {model_name} not found")