class Tokenizer():
    """
        <<Interface>>\n
        Functions:
            - encode()
            - count_tokens()
    """
    
    def encode(self) -> None:
        raise NotImplementedError("The 'Encode' function must to be implemented")
        
    def count_tokens(self) -> None:
        raise NotImplementedError("The 'Count_Tokens' function must to be implemented")