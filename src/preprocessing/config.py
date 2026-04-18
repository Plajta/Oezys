import configparser
import json
import os


class Config:
    def __init__(self):
        conf = configparser.ConfigParser()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_dir)
        repo_root = os.path.dirname(src_dir)
        conf_path = os.path.join(current_dir, "config.conf")
        
        conf.read(conf_path)

        self.source_url = conf["preprocessing"]["source_url"]
        
        # Lock paths absolutely relative to the repo's root folder
        self.raw_dir = os.path.join(repo_root, conf["preprocessing"]["raw_dir"])
        self.clean_dir = os.path.join(repo_root, conf["preprocessing"]["clean_dir"])
        self.exp_dir = os.path.join(repo_root, conf["preprocessing"]["exp_dir"])
        self.classes_raw_dirs = json.loads(conf.get("preprocessing", "classes_raw_dirs"))


CONFIG = Config()