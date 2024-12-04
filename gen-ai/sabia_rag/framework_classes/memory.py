from framework_classes.message import Message

class Memory:
    """
        Responsible for history management (context).\n
        Functions:
            - add_message(): add a message to the history.
            - get_history(): access history.
    """
    
    def __init__(self, limit: int = None) -> None:
        """Creates an instance with a specified history length."""
        self.history = []
        self.limit = limit
    
    def add_message(self, message: Message) -> None:
        """
            Add a message to the history.\n
            Parameter:
                A Message object
            Return:
                None
        """
        if not isinstance(message, Message):
            raise ValueError("Invalid Message")

        self.history.append(message)
        if self.limit != None and len(self.history) > self.limit:
            self.history = self.history[-self.limit:]

    def get_history(self) -> list[Message]:
        """
            Access the history.\n
            Parameter:
                None
            Return:
                List of Message objects that make up the history
        """
        return self.history 