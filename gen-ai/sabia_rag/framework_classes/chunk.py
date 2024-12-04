from typing import Any

class Chunk:
    """
        Representation of Chunk structure. Consisting of:
            - content: a text string (example: user input).
            - metadata: a dictionary of source info (example: filename)
            - index: number for chunks ordering. Used to reconstruct a Doc from its chunks.
    """

    def __init__(self, content: str, metadata: dict, index: int = None) -> None:
        """Creates an instance with a content, metadata and if necessary index."""
        self.content = content
        self.index = index
        self.metadata = metadata