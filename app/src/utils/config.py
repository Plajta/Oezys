import os
from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("type", "dev")
APP_NAME = "RadBrecim"
APP_VERSION = "0.1.0"

IS_DEV = APP_ENV == "dev"
