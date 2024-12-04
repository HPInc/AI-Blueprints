from framework_classes.vectordb import VectorDB
from framework_classes.chain import Chain
from framework_classes.memory import Memory
from framework_classes.message import Message
from src.models.demo_model import DemoModel
from src.prompt_templates.demo_prompt_template import DemoPromptTemplate
from src.prompts.demo_prompt import DemoPrompt
import time

class DemoQueryChain(Chain):
    """
        Specialization of a `Chain`.\n
        Functions:
            - start(): continuously runs DemoQueryChain cycles.
            - run(): run a single DemoQueryChain cycle.
    """
    
    def __init__(self, vectordb: VectorDB=None) -> None:
        """Creates an instance with a VectorDB (optional)."""
        self.model = DemoModel(n_gpu_layers=~10, stop_tags=["</answer>"])
        self.db = vectordb

        prompt_template = DemoPromptTemplate()
        self.prompt = DemoPrompt(prompt_template, self.db)

        self.memory = Memory(20)
    
    
    def start(self) -> None:
        """
            Continuously runs DemoQueryChain cycles.\n
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
            Run a single DemoQueryChain cycle.\n
            Parameter:
                A user text string
            Return:
                Model response to user input (a text string)
        """

        message = Message("User", message)
        history = self.memory.get_history()        
        content = self.prompt.get_prompt(message, history)
        prediction = self.model.predict(content)+" "
        start = prediction.index("Answer: <answer>")+16
        prediction = prediction[start:prediction.find("User", start)].strip()
        self.memory.add_message(message)
        self.memory.add_message(Message("Assistant", prediction))
        return prediction

 