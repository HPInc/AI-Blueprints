from framework_classes.prompt import Prompt
from framework_classes.prompt_template import PromptTemplate
from typing import Any

class DynamicPrompt(Prompt):
    """
        Specialization of a `Prompt`.\n
        Functions:
            - get_prompt(): apply the given inputs to a template.
    """

    def __init__(self, template: PromptTemplate) -> None:
        """Creates an instance with a specified template."""
        self.template = template

    def get_prompt(self, values: dict(str, Any)) -> str:
        """
            Apply the given inputs to the template.\n
            Parameter:
                A dictionary with position keys for
                the template to apply its values
            Return:
                A text string using the template with the inputs
        """
        return self.template.apply(values)