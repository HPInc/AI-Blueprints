import threading
from framework_classes.chain import Chain

class ComposedChain(Chain):
    """
        Responsible for ...\n
        Functions:
            - invoke(): start the chain function received in the new thread.
            - summon(): runs multiple Chains on different threads.
    """

    def invoke(self, name: str, function, params: tuple=None) -> None:
        """
            Start the chain function received in the new thread.\n
            Parameter:
                Name to identify the thread and the function with the respective params
            Return:
                None
        """

        if params == None:
            function()
        else:
            function(*params)

    def summon(self, chains: list[Chain]) -> None:
        """
            Runs multiple Chains on different threads.\n
            Parameter:
                A list of chains
            Return:
                None
        """

        for chain in chains:
            threading.Thread(
                target=self.invoke,
                name=chain[0],
                args=chain
            ).start()