# *** AI Studio experiment test ***
# notebook:
#  path: "run-workflow.ipynb"
#  class_name: CovidNotebook
# workspaces:
#  - datascience
#  - datasciencegpu
#  - deeplearning
#  - deeplearninggpu
# ******

import unittest


class TestCovidNotebook(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.notebook = CovidNotebook()
        cls.notebook.run()

    # Verifies if the notebook runs without errors
    def test_notebook_run(self):
        self.assertTrue(True, "Notebook did not run successfully")


if __name__ == "__main__":
    unittest.main()
