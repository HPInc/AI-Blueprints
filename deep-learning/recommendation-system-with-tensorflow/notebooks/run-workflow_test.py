# *** AI Studio experiment test ***
#notebook:
#  path: "run-workflow.ipynb"
#  class_name: RecommendationSystemNotebook
#  variables:
#   - movie_titles
#   - testing_flag
#workspaces:
#  - deeplearning
#  - deeplearninggpu
# ******

import unittest

class TestRecommendationNotebook(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.notebook = RecommendationSystemNotebook()
        cls.notebook.testing_flag = True
        cls.notebook.run()
    
    # Verifies if the notebook runs without errors
    def test_notebook_run(self):
        self.assertTrue(True, "Notebook did not run successfully")

if __name__ == '__main__':
    unittest.main()
