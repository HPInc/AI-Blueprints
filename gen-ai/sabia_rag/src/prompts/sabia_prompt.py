from framework_classes.prompt import Prompt
from framework_classes.prompt_template import PromptTemplate
from framework_classes.message import Message
from framework_classes.memory import Memory

class SabIAPrompt(Prompt):
    """
        Specialization of a `Prompt`.\n
        Functions:
            - get_prompt(): apply the given inputs to a template.
    """

    def __init__(self, template: PromptTemplate) -> None:
        """Creates an instance with a specified template."""
        self.template = template

    def get_prompt(self, message: Message, history: Memory) -> str:
        """
            Apply the given inputs to the template.\n
            Parameter:
                A Message object with the user's text string
                and Message history
            Return:
                A text string using the template with the inputs
        """
        return self.template.apply(message=message, history=history)