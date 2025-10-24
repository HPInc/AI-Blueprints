import os, shutil
import re
import argparse
from nbconvert import PythonExporter
from filename_utils import FilenameUtils

import ast
import astor  # pip install astor


class VariableRenamer(ast.NodeTransformer):
    """AST node transformer for renaming variables to instance variables."""

    def __init__(self, variables=[]):
        """
        Initialize the variable renamer.

        Args:
            variables (list): List of variable names to rename with 'self.' prefix
        """
        self.variables = variables

    def visit_Name(self, node):
        """
        Visit Name nodes in the AST and rename variables if they're in the variables list.

        Args:
            node: AST Name node

        Returns:
            Modified AST node with renamed variable
        """
        # Rename variable references
        if node.id in self.variables:
            node.id = f"self.{node.id}"
        return node


class CodePreprocessor:
    """Preprocesses code files, converting notebooks to Python and encapsulating them in classes."""

    def __init__(self, metadata, destination, logger=None):
        """
        Initialize the code preprocessor.

        Args:
            metadata (dict): Metadata containing code folder, test info, and notebook configurations
            destination (str): Destination folder for processed files
            logger: Optional logger instance for debugging
        """
        def_ignore = os.path.join(os.path.dirname(__file__), "ignore_list.txt")
        self.source = metadata.get("code_folder", ".")
        self.destination = destination
        self.logger = logger
        self.ignore_file = def_ignore
        self.code_info = metadata.get("code", {})
        self.fnu = FilenameUtils()
        self.renamer = VariableRenamer()
        with open(self.ignore_file, "r") as file:
            self.ignore_list = file.readlines()
        self.process_all_files(self.source, self.destination)

    def rename_variables_in_code(self, variables, source_code):
        """
        Rename variables in the given source code using AST transformation.

        Args:
            variables (list): List of variable names to rename with 'self.' prefix
            source_code (str): Python source code to process

        Returns:
            str: Modified source code with renamed variables
        """
        try:
            tree = ast.parse(source_code)
            renamer = VariableRenamer(variables=variables)
            new_tree = renamer.visit(tree)
            return astor.to_source(new_tree)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error renaming variables: {e}")
            return source_code

    def process_notebook(self, src_path, dst_path):
        """
        Convert a Jupyter notebook to a Python file.

        Args:
            src_path (str): Path to the source notebook file
            dst_path (str): Path where the converted Python file will be saved
        """
        exporter = PythonExporter()
        python_code = exporter.from_filename(src_path)
        with open(dst_path, "w") as dst_file:
            dst_file.write(python_code[0])

    def encapsulate_code(self, nb_info, lines):
        """
        Encapsulate notebook code in a class with a run method.

        Preserves imports and comments at the top level, then wraps the rest
        of the code in a class with the specified name and a run() method.

        Args:
            nb_info (dict): Notebook information containing class_name
            lines (list): List of code lines to encapsulate

        Returns:
            list: Modified lines with proper class encapsulation
        """
        new_lines = []
        before_class = True
        for line in lines:
            if before_class:
                if line.strip() == "" or line.strip()[0] == "#" or "import" in line:
                    new_lines.append(line)
                else:
                    new_lines.append("\n")
                    new_lines.append(f"class {nb_info['class_name']}():\n")
                    new_lines.append("  def run(self):\n")
                    new_lines.append(f"    {line}\n")
                    before_class = False
            else:
                new_lines.append(f"    {line}\n")
        return new_lines

    def process_python_file(self, src_path, dst_path, nb_info=None):
        """
        Process a Python file by removing ignored patterns and optionally encapsulating in a class.

        Args:
            src_path (str): Path to the source Python file
            dst_path (str): Path where the processed file will be saved
            nb_info (dict, optional): Notebook info for encapsulation. If provided,
                                    the code will be wrapped in a class
        """
        with open(src_path, "r") as f:
            lines = f.readlines()
        for expression in self.ignore_list:
            lines = [line for line in lines if not re.match(expression.strip(), line)]
        if nb_info is not None:
            lines = self.rename_variables_in_code(
                nb_info.get("variables", []), "".join(lines)
            )
            lines = self.encapsulate_code(nb_info, lines.split("\n"))
            lines = [
                line + "\n" for line in lines
            ]  # Ensure each line ends with a newline character
        with open(dst_path, "w") as outfile:
            outfile.write("".join(lines))

    def get_nb_info(self, full_name):
        """
        Get notebook information for a given file path.

        Args:
            full_name (str): Full path of the notebook file

        Returns:
            dict or None: Notebook configuration if found, None otherwise
        """
        for _, test_metadata in self.code_info.items():
            if ("notebook" in test_metadata) and ("path" in test_metadata["notebook"]):
                fixed_notebook_path = self.fnu.fix_filename(
                    test_metadata["notebook"]["path"]
                )
                fixed_full_name = self.fnu.fix_filename(full_name)
                if fixed_notebook_path == fixed_full_name:
                    return test_metadata["notebook"]
        return None

    def process_all_files(self, src_path, dst_path, internal=False):
        """
        Recursively process all files in a directory structure.

        Handles different file types:
        - Jupyter notebooks: Converts to Python and encapsulates in classes
        - Python files: Processes and copies
        - Other files: Copies as-is

        Args:
            src_path (str): Source directory path
            dst_path (str): Destination directory path
            internal (bool): Whether this is an internal recursive call

        Returns:
            str: Status message indicating the result of processing
        """
        dst_path = self.fnu.fix_filename(dst_path, extension="")
        if not os.path.isdir(src_path):
            return "Not a folder"
        if os.path.basename(src_path)[0] == ".":
            if internal:
                return "Ignored folder"
        files = os.listdir(src_path)
        if os.path.exists(dst_path):
            if not os.path.isdir(dst_path):
                return "Not a folder"
        else:
            os.mkdir(dst_path)
        with open(os.path.join(dst_path, "__init__"), "w") as init_file:
            init_file.write("")
        for file in files:
            src_full = os.path.realpath(os.path.join(src_path, file))
            if os.path.isdir(src_full):
                self.process_all_files(
                    src_full, os.path.join(dst_path, file), internal=True
                )
            else:
                if self.fnu.filetype(src_full) == "notebook":
                    if self.get_nb_info(src_full) is not None:
                        tmp_filename = os.path.join(
                            dst_path, self.fnu.temp_python_name(file)
                        )
                        self.process_notebook(src_full, tmp_filename)
                        self.process_python_file(
                            tmp_filename,
                            os.path.join(dst_path, self.fnu.fix_filename(file)),
                            self.get_nb_info(src_full),
                        )
                        os.remove(tmp_filename)
                elif self.fnu.filetype(src_full) == "python":
                    self.process_python_file(
                        src_full, os.path.join(dst_path, self.fnu.fix_filename(file))
                    )
                else:
                    shutil.copyfile(src_full, os.path.join(dst_path, file))
        return "Files processed"
