# *** AI Studio experiment test ***
#notebook:
#  path: "generative-ai/vanilla-rag-with-langchain/notebooks/vanilla-rag-with-langchain-and-galileo.ipynb"
#  class_name: GenAiChatbotNotebook
#workspaces:
#  - rapidsbase
#  - rapidsnotebook
# ******

import unittest

class TestGenAiChatbotNotebook(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.notebook = GenAiChatbotNotebook()
        cls.notebook.run()
       
if __name__ == '__main__':
    unittest.main()