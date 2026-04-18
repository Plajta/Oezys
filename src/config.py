import configparser
import json
import os

class Config:
    def __init__(self):
        conf = configparser.ConfigParser()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        conf_path = os.path.join(current_dir, "config.conf")
        
        conf.read(conf_path)

        self.source_url = conf["preprocessing"]["source_url"]
        self.raw_dir = conf["preprocessing"]["raw_dir"]
        self.clean_dir = conf["preprocessing"]["clean_dir"]
        self.classes_raw_dirs = json.loads(conf.get("preprocessing", "classes_raw_dirs"))


CONFIG = Config()