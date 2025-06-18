# *** AI Studio experiment test ***
#notebook:
#  path: "ngc-integration/stock_analysis_with_pandas_and_cuDF/notebooks/stock_analysis_with_pandas_and_cuDF.ipynb"
#  class_name: StocksCudfNotebook
#workspaces:
#  - rapidsbase
#  - rapidsnotebook
# ******

import unittest

class TestStocksCudfNotebook(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.notebook = StocksCudfNotebook()
        cls.notebook.run()
       
if __name__ == '__main__':
    unittest.main()