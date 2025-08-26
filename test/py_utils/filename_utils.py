import os
from datetime import datetime

class FilenameUtils():
    def __init__(self):
        self.chars_to_replace = [" ", "[", "]", "{", "}", "-"]

    #This function identify the type of the file being preprocessed
    def filetype(self, filename):
        _, extension = os.path.splitext(filename)
        if extension.lower() == ".py":
            return "python"
        elif extension.lower() == ".ipynb":
            return "notebook"
        else:
            return "other"

    #This fixes the name of the file, removing undesirable chars and passing to lower case
    def fix_filename(self, filename, extension = ".py"):
        filename = f"{os.path.splitext(filename)[0]}{extension}"
        for char in self.chars_to_replace:
            filename = filename.replace(char, "_")
        return filename.lower()  
    
    #This method creates a name for a temporary python file
    def temp_python_name(self, filename):
        basename = os.path.basename(filename)
        basename, _= os.path.splitext(basename)
        return f"tmp_{basename}.py"
    
    #This method will return the module name from the filename
    def get_module(self, filename):
        folder_list = os.path.dirname(filename).split(os.sep)
        folder_list.append(os.path.splitext(os.path.basename(filename))[0])
        fixed_folders = [self.fix_filename(folder, extension = "") for folder in folder_list]
        fixed_folders = [folder for folder in fixed_folders if folder != ""]
        return ".".join(fixed_folders)

    def create_empty_folder(self, dirname):
        n = datetime.now()
        backup_suffix = f"bck{n.year}{n.month}{n.day}-{n.hour}{n.minute}{n.second}"
        if os.path.exists(dirname):
            os.rename(dirname, f"{dirname}_{backup_suffix}")
        os.mkdir(dirname)
        
            