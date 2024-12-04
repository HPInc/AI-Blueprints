class Document:
    """
        Representation of a Document object. Consisting of:
            - metadata: a dictionary containing the document attributes.
            - content: a text string.
    """
    def __init__(self, content: str, metadata: dict):
        """Creates an instance with content and if necessary metadata."""
        self.content = content
        self.metadata = metadata