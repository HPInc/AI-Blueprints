from framework_classes.chain import Chain
from src.models.sabia_model import SabIAModel

class SabIAChain(Chain):
    """
        Specialization of a `Chain`.\n
        Functions:
            - start(): continuously runs SabIAChain cycles.
            - run(): run a single SabIAChain cycle.
    """
    
    def __init__(self) -> None:
        """Creates an instance without arguments."""
        super().__init__()
        self.model = SabIAModel()
    
    def start(self) -> None:
        """
            Continuously runs SabIAChain cycles.\n
            Parameter:
                None
            Return:
                None
        """
        print("\nSabIA: Hi! I'm SabIA. How can I help you?")
        while True:
            message = input("\nUser: ")
            self.run(message)

    def run(self, message: str) -> str:
        """
            Run a single SabIAChain cycle.\n
            Parameter:
                A user text string
            Return:
                Model response to user input (a text string)
        """
        self.model.predict(message)