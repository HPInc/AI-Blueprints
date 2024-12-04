from framework_classes.prompt_template import PromptTemplate
from framework_classes.message import Message
from typing import Any

class DynamicPromptTemplate(PromptTemplate):
    """
        Specialization of a `PromptTemplate`.\n
        Functions:
            - apply(): apply the given inputs to a pattern.
            - format_content(): formats an input into a text string.
    """

    def __init__(self, pattern: str) -> None:
        """Creates an instance with a specified pattern."""
        self.pattern = pattern
        
    def apply(self, values: dict(str, Any)) -> str:
        """
            Apply the given inputs to a pattern.\n
            Parameter:
                A dictionary with position keys in the pattern
                and values to replace them
            Return:
                A text string of the pattern with the inputs
        """
        content = self.pattern

        for key, value in values.items():
            content = content.replace("{"+key+"}", self.format_content(value))
            
        return content
    
    def format_content(self, content: Any) -> str:
        """
            Formats the value associated with a key in the
            dictionary to a text string.\n
            Parameter:
                Any
            Return:
                A text string representing the input
        """
        if isinstance(content, str):
            return content
        if isinstance(content, Message):
            return content.role+": "+content.content
        if isinstance(content, list):
            return "\n".join([self.format_content(x) for x in content])
        return str(content)
