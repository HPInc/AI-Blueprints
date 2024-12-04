from framework_classes.model import Model

class EmbeddingsModel(Model):
    """
        <<Interface>>\n
        Functions:
            - embed()
    """

    def embed(self) -> None:
        raise NotImplementedError("The 'Embed' function must to be implemented")