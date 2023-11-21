import os


def create_dir(dir_path: str) -> None:
    try:
        os.makedirs(dir_path)
    except:
        pass
