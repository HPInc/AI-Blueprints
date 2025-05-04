import os, shutil
import argparse
from nbconvert import PythonExporter
from filename_utils import FilenameUtils

class CodePreprocessor():
    def __init__(self, metadata, destination, logger = None):
        def_ignore = os.path.join(os.path.dirname(__file__), "ignore_list.txt")
        self.source = metadata.get("code_folder", ".")
        self.destination = destination
        self.logger = logger
        self.ignore_file = def_ignore
        self.code_info = metadata.get("code", {})
        self.fnu = FilenameUtils()
        with open(self.ignore_file, "r") as file:
            self.ignore_list = file.readlines()
        self.process_all_files(self.source, self.destination)

    #This method converts a notebook into a temporary python file
    def process_notebook(self, src_path, dst_path):
        exporter = PythonExporter()
        python_code = exporter.from_filename(src_path)
        with open(dst_path, "w") as dst_file:
            dst_file.write(python_code[0])
            
    def process_code_line(self, nb_info, line):
        for variable in nb_info["variables"]:
            line = line.replace(variable, f"self.{variable}")
        return f"    {line}" 
        
    #This method adds the proper class and run method to encapsulate the notebook code
    def encapsulate_code(self, nb_info, lines):
        new_lines = []
        before_class = True
        for line in lines:
            if before_class:
                if line.strip() == "" or line.strip()[0] == '#' or "import" in line:
                    new_lines.append(line)
                else:
                    new_lines.append("\n")
                    new_lines.append(f"class {nb_info['class_name']}():\n")
                    new_lines.append("  def run(self):\n")
                    new_lines.append(self.process_code_line(nb_info, line))
                    before_class = False
            else:
                new_lines.append(self.process_code_line(nb_info, line))
        return new_lines

    def process_python_file(self, src_path, dst_path, nb_info = None):
        with open(src_path, "r") as f:
            lines = f.readlines()
        for expression in self.ignore_list:
            lines = [line for line in lines if expression.strip() not in line]
        if nb_info is not None: 
            lines = self.encapsulate_code(nb_info, lines)
            with open(dst_path, "w") as outfile:
                outfile.write("".join(lines))
    
    def get_nb_info(self, full_name):
        for _, test_metadata in self.code_info.items():
            if ("notebook" in test_metadata) and ("path" in test_metadata["notebook"]):
                if test_metadata["notebook"]["path"] == full_name:
                    return test_metadata["notebook"]
        return None
    
    def process_all_files(self, src_path, dst_path, internal = False):
        dst_path = self.fnu.fix_filename(dst_path, extension = "")
        if not os.path.isdir(src_path):
            return "Not a folder"
        if os.path.basename(src_path)[0] == ".":
            if internal:
                return "Ignored folder"
        files = os.listdir(src_path)
        if os.path.exists(dst_path):
            if not os.path.isdir(dst_path):
                return("Not a folder")
        else:
            os.mkdir(dst_path)
        for file in files:
            src_full = os.path.relpath(os.path.join(src_path, file))
            if os.path.isdir(src_full):             
                self.process_all_files(src_full, os.path.join(dst_path, file), internal = True)
            else:
                if self.fnu.filetype(src_full) == "notebook":
                    if self.get_nb_info(src_full) is not None:
                        tmp_filename = os.path.join(dst_path, self.fnu.temp_python_name(file))
                        self.process_notebook(src_full, tmp_filename)
                        self.process_python_file(tmp_filename,
                                                os.path.join(dst_path, self.fnu.fix_filename(file)), 
                                                self.get_nb_info(src_full))
                        os.remove(tmp_filename)
                elif self.fnu.filetype(src_full) == "python":
                    self.process_python_file(src_full, os.path.join(dst_path, self.fnu.fix_filename(file)))
                else:
                    shutil.copyfile(src_full, os.path.join(dst_path, file))
        return ("Files processed")
                
    