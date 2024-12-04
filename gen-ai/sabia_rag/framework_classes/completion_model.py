from framework_classes.model import Model

class CompletionModel(Model):
    """
        <<Interface>>\n
        Functions:
            - predict()
    """
    
    def predict(self) -> None:
        raise NotImplementedError("The 'Predict' function must to be implemented")