# *** AI Studio experiment test ***
#notebook:
#  path: "run-workflow.ipynb"
#  class_name: MnistNotebook
#  variables:
#  - x_train
#  - x_test
#  - model
#workspaces:
#  - deeplearning
#  - deeplearninggpu
# ******

import unittest

class TestMnistNoteboook(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.notebook = MnistNotebook()
        cls.notebook.run()

    #Verifies if the training set has more than 50k entries
    def test_training_size(self):
        self.assertGreaterEqual(len(TestMnistNoteboook.notebook.x_train), 50000)
    
    #Verifies if the test set has at least 10k entries
    def test_validation_size(self):
        self.assertGreaterEqual(len(TestMnistNoteboook.notebook.x_test), 10000)
        
if __name__ == '__main__':
    unittest.main()