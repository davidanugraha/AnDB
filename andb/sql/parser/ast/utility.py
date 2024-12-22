from .base import ASTNode

class Command(ASTNode):
    def __init__(self, command: str, parameters: dict):
        super().__init__()
        self.command = command
        self.parameters = parameters
    