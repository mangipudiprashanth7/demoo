from importlib.resources import path
from os import path
UPLOAD_FOLDER = path.join(path.dirname(path.abspath(__file__)), "static/uploads/")
print(UPLOAD_FOLDER)
print(path.dirname(path.abspath(__file__)))