class Message:
    """
        Representation of historical messages. Consisting of:
            - role: a text string responsible for the content (Assistant, User or System).
            - content: a text string (example: user input).
            - tokens_info: a dictionary
            (key: tokenizer encoding name, value: number of content tokens). Default None.
    """

    ROLES = ["Assistant", "System", "User"]
    
    def __init__(self, role: str, content: str, tokens_info: dict = None) -> None:
        """Creates an instance with a role, content and if necessary tokens_info."""
        if role not in self.ROLES:
            raise ValueError("Invalid role")
        self.role = role

        self.content = content
        
        if not isinstance(tokens_info, dict) and not tokens_info == None:
            raise ValueError("Tokens_info must be a dictionary or none")
        self.tokens_info = tokens_info