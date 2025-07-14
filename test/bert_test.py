# *** AI Studio experiment test ***
#notebook:
#  path: "deep-learning/question-answering-with-bert/notebooks/question_answering_system__with_BERT.ipynb"
#  class_name: BertNotebook
#workspaces:
#  - deeplearning
#  - deeplearninggpu
# ******

import unittest

class TestBertNotebook(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.notebook = BertNotebook()
        cls.notebook.run()
    
    # Verifies if the notebook runs without errors
    def test_notebook_run(self):
       self.assertTrue(True, "Notebook did not run successfully")

if __name__ == '__main__':
    unittest.main()