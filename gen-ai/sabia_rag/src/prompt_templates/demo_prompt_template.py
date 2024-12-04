from framework_classes.prompt_template import PromptTemplate
from framework_classes.message import Message
from framework_classes.memory import Memory
from framework_classes.chunk import Chunk

class DemoPromptTemplate(PromptTemplate):
    """
    Specialization of a PromptTemplate.
    Functions:
        - _init_(): Initializes the class with a customizable pattern.
        - apply(): Apply the given inputs to the class pattern.
    """

    default_pattern = """The Assistant have the following informations to reply about the data science platform:
<information>
{background}
</information>

The following conversation is from the Assistant especialist in data science and an user from a data science platform. Complete what the Assistant would reply without ask anything or any other comments.
<conversation>
<Assistant>Hi!</Assistant>
{context}
</conversation>
<question>
<User>{question}</User>
</question>
Answer: <answer>"""

    def __init__(self, pattern=None):
        """
        Initializes the instance with a specific pattern or the default pattern if none is provided.
        """
        self.pattern = pattern if pattern else self.default_pattern

    def apply(self, message: Message, history: Memory, background: list[Chunk]) -> str:
        """
        Apply the given inputs to the class pattern.
        Parameter:
            A Message object with the user's text string, Message history
            and the list of chunks found by VectorDB (similarity search)
        Return:
            A text string already using the pattern with the inputs
        """
        
        background = "\n".join(['"'+x.content+'"' for x in background])
        context = "\n".join(["<"+x.role+">"+x.content+"</"+x.role+">" for x in history])

        content = self.pattern.replace("{background}", ""+background+"" if background else "")
        content = content.replace("{context}", context)
        content = content.replace("{question}", message.content)

        # Prints the built prompt for diagnostics
        print("Prompt:", content)

        return content