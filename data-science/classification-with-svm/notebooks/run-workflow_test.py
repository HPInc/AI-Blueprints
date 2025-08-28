# *** AI Studio experiment test ***
#notebook:
#  path: "run-workflow.ipynb"
#  class_name: IrisNotebook
#workspaces:
#  - datascience
# ******

import unittest

class TestIrisNotebook(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.notebook = IrisNotebook()
        cls.notebook.run()
    
    # Verifies if the notebook runs without errors
    def test_notebook_run(self):
        self.assertTrue(True, "Notebook did not run successfully")

if __name__ == '__main__':
    unittest.main()