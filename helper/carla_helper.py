import collections
import os
import re
import signal
import sys

import carla
import cv2
import numpy as np
import psutil
from PIL import Image

from helper.list_procs import search_procs_by_name


def update_config(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update_config(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def add_carla_path(carla_path_config_file):
    carla_text_path = (os.path.dirname(os.path.realpath(__file__)) + "/" + carla_path_config_file)
    carla_path_file = open(carla_text_path, "r")
    carla_main_path = (carla_path_file.readline().split("\n"))[0]
    carla_path_file.close()
    for file in os.listdir(carla_main_path + "/PythonAPI/carla/dist/"):
        if 'py3.7' in file:
            carla_egg_file = os.path.join(carla_main_path + "/PythonAPI/carla/dist/", file)
    sys.path.append(os.path.expanduser(carla_egg_file))
    carla_python_interface = carla_main_path + "/PythonAPI/carla/"
    carla_server_binary = carla_main_path + "/CarlaUE4.sh"
    sys.path.append(os.path.expanduser(carla_python_interface))
    print(carla_python_interface)
    return carla_server_binary


def get_parent_dir(directory):
    return os.path.dirname(directory)


def post_process_image(image, normalized=True, grayscale=True):
    """Convert image to gray scale and normalize between -1 and 1 if required"""
    if isinstance(image, list):
        image = image[0]
    
    if grayscale:
        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).convert("L"))
        image = image[:, :, np.newaxis]
    
    if normalized:
        return (image.astype(np.float32) - 128) / 128
    else:
        return image.astype(np.uint8)


def kill_server():
    """Kill all PIDs that start with Carla. Do this if you running a single server"""
    for pid, name in search_procs_by_name("Carla").items():
        os.kill(pid, signal.SIGTERM)


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]