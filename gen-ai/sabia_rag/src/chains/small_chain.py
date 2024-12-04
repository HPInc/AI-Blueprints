from framework_classes.chain import Chain
from framework_classes.memory import Memory
from framework_classes.message import Message
from src.models.sabia_model import SabIAModel
from src.prompt_templates.sabia_prompt_template import SabIAPromptTemplate
from src.prompts.sabia_prompt import SabIAPrompt

class SmallChain(Chain):
    """
        Specialization of a `Chain`.\n
        Functions:
            - start(): continuously runs SmallChain cycles.
            - run(): run a single SmallChain cycle.
    """
    
    def __init__(self) -> None:
        """Creates an instance without arguments."""
        self.model = SabIAModel()
        prompt_template = SabIAPromptTemplate()
        self.prompt = SabIAPrompt(prompt_template)
        self.memory = Memory(10)

    def start(self) -> None:
        """
            Continuously runs SmallChain cycles.\n
            Parameter:
                None
            Return:
                None
        """
        print("\nSabIA: Hi! I'm SabIA. How can I help you?")
        while True:
            message = input("\nUser: ")
            print("\nSabIA: "+self.run(message))

    def run(self, message: str) -> str:
        """
            Run a single SmallChain cycle.\n
            Parameter:
                A user text string
            Return:
                Model response to user input (a text string)
        """
        message = Message("User", message)
        history = self.memory.get_history()        
        content = self.prompt.get_prompt(message, history)

        prediction = self.model.predict(content)
        prediction = prediction[prediction.rindex("Answer: ")+8:]
        self.memory.insert_message(message)
        self.memory.insert_message(Message("Assistant", prediction))
        return prediction