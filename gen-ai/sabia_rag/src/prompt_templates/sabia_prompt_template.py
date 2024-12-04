from framework_classes.prompt_template import PromptTemplate
from framework_classes.message import Message
from framework_classes.memory import Memory

class SabIAPromptTemplate(PromptTemplate):
    """
        Specialization of a `PromptTemplate`.\n
        Functions:
            - apply(): apply the given inputs to the class pattern.
    """

    pattern = """Context:\n{context}\nQuestion: {question}\nAnswer: """

    def apply(self, message: Message, history: Memory) -> str:
        """
            Apply the given inputs to the class pattern.\n
            Parameter:
                A Message object with the user's text string
                and Message history
            Return:
                A text string already using the pattern with the inputs
        """
        context = "\n".join([x.role+": "+x.content for x in history])
        content = self.pattern.replace("{context}", context)
        content = content.replace("{question}", message.content)
        return content
