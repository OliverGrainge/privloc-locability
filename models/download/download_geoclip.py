from models.geolocation import load_model 


def main(): 
    load_model("geoclip")
    return True 

if __name__ == "__main__": 
    main()