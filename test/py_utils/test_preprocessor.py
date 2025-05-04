import os, shutil
import yaml
from filename_utils import FilenameUtils

class TestPreprocessor:
    def __init__(self, test_script, target_folder, workspace = "minjp", logger = None):
        self.begin_mark = "***"
        self.end_mark = "***"
        self.logger = logger
        with open(test_script) as file:
            self.metadata = yaml.safe_load(file)
        self.metadata["code"] = {}
        self.fnu = FilenameUtils()
        self.target_folder = target_folder
        self.workspace = workspace
        self.fnu.create_empty_folder(target_folder)
        with open(os.path.join(target_folder, "__init__"), "w") as init_file:
            init_file.write("")
                
    def preprocess_test_file(self, base_folder, file_name):
        with open(os.path.join(base_folder, file_name), "r") as file:
            lines = file.readlines()
        started_yaml = False
        yaml_lines = []
        next_line = 0
        for index, line in enumerate(lines):
            next_line += 1
            if not started_yaml:
                if self.begin_mark in line:
                    started_yaml = True
            else:
                if self.end_mark in line:
                    break
                else:
                    yaml_lines.append(line[1:])
        file_metadata = yaml.safe_load("".join(yaml_lines))
        if "workspaces" not in file_metadata:
            return None
        if self.workspace not in file_metadata["workspaces"]:
            return None
        if "notebook" in file_metadata:
            notebook_path = file_metadata["notebook"]["path"]
            class_name = file_metadata["notebook"]["class_name"]
            import_line = f"from {self.fnu.get_module(notebook_path)} import {class_name} \n"
            lines.insert(next_line, import_line)
        with open(os.path.join(self.target_folder, file_name), "w") as outfile:
            outfile.write("".join(lines))
        self.metadata["code"][file_name] = file_metadata
        
    def run(self):
        base_folder = self.metadata["test_folder"]
        tests = self.metadata["files"]
        for test in tests:
            self.preprocess_test_file(base_folder, test)
        
        
           
        