# *** AI Studio experiment test ***
#notebook:
#  path: "deep-learning/question_answering_with_BERT/notebooks/question_answering_system__with_BERT.ipynb"
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
       
if __name__ == '__main__':
    unittest.main()