import os


def root_dir():
    return os.path.dirname(os.path.realpath(__file__))


def top_dir():
    avod_root_dir = root_dir()
    return os.path.split(avod_root_dir)[0]
