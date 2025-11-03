import os, shutil
import argparse
from nbconvert import PythonExporter
from code_preprocessor import CodePreprocessor
from test_preprocessor import TestPreprocessor
import logging

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser("Preprocessor to allow unit tests")
    parser.add_argument(
        "-s", "--script", help="YAML file with the test folder and files"
    )
    # parser.add_argument("-i", "--input", help="Input folder with the test files")
    parser.add_argument(
        "-o", "--output", help="Destination folder to keep the processed files"
    )
    parser.add_argument("-w", "--workspace", help="Workspace used to load the files")
    parser.add_argument(
        "--ignore",
        help="Text file with list of expressions to be ignored",
        default=os.path.join(os.path.dirname(__file__), "ignore_list.txt"),
    )
    args = parser.parse_args()
    logger.info(
        f"Parsed the arguments: output: {args.output}, workspace: {args.workspace}"
    )
    test_preprocessor = TestPreprocessor(
        args.script, args.output, workspace=args.workspace, logger=logger
    )
    test_preprocessor.run()
    logger.info("Preprocessed the tests")
    code_preprocessor = CodePreprocessor(
        test_preprocessor.metadata, args.output, logger=logger
    )
    logger.info("Preprocessed the code")
    test_preprocessor.copy_requirements()
