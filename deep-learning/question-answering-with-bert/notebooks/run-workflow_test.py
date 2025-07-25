# *** AI Studio experiment test ***
#notebook:
#  path: "run-workflow.ipynb"
#  class_name: BertNotebook
#  variables:
#   - squad_dataset
#   - testing_flag
#workspaces:
#  - deeplearning
#  - deeplearninggpu
#requirements: "../requirements.txt"
# ******

import unittest

class TestBertNotebook(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.notebook = BertNotebook()
        cls.notebook.testing_flag = True
        cls.notebook.run()
    
    # Verifies if the notebook runs without errors
    def test_notebook_run(self):
        self.assertTrue(True, "Notebook did not run successfully")

    #Verifies if the Squad dataset is loaded
    def test_squad_dataset_loaded(self):
        self.assertTrue(TestBertNotebook.notebook.squad_dataset is not None, "Squad dataset is not loaded")

if __name__ == '__main__':
    unittest.main()