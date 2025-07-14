
# *** AI Studio experiment test ***
#notebook:
#  path: "AI-Blueprints"/data-science/classification-with-svm/notebooks/run-workflow.ipynb"
#  class_name: IrisNotebook
#workspaces:
#  - data science
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