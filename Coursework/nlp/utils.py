import os
import abc


class Utils(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        ()

    def create_folder(self, folder_path: str) -> None:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
