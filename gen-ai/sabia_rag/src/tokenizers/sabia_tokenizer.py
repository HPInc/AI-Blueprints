from framework_classes.tokenizer import Tokenizer
import tiktoken

class SabIATokenizer(Tokenizer):
    """
        Specialization of a `Tokenizer` that uses the tiktoken library.\n
        Functions:
            - encode(): generate tokens from a string.
            - count_tokens(): counts the number of tokens in a string.
    """
    
    def __init__(self, encoding_name:str = "cl100k_base") -> None:
        """Creates an instance with an encoding name for the tokenizer. Default: `cl100k_base`"""
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def encode(self, text: str) -> list[str]:
        """
            Generate tokens from a string.\n
            Parameter:
                A text string
            Return:
                A list of tokens from a text string
        """
        tokens = self.encoding.encode(text)
        return [(self.encoding.decode([token])) for token in tokens]
    
    
    def count_tokens(self, text: str) -> int:
        """
            Counts the number of tokens in a string.\n
            Parameter: 
                A text string
            Return:
                The number of tokens in a text string
        """
        return len(self.encoding.encode(text))