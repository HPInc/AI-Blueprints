class Chain:
    """
        <<Interface>>\n
        Functions:
            - start()
            - run()
    """

    def start(self) -> None:
        self.run()

    def run(self) -> None:
        raise NotImplementedError("The 'Run' function must to be implemented")