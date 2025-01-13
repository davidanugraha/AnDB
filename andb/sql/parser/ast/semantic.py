from .base import ASTNode

class FileSource(ASTNode):
    def __init__(self, file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.parts = file_path.value
        
class Prompt(ASTNode):
    def __init__(self, prompt_text, defined_column, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_text = prompt_text
        self.defined_column = defined_column
        self.alias = None
        
class SemanticTabular(ASTNode):
    def __init__(self, identifier, expr_list, table_source, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_schemas = expr_list
        self.table_source = table_source
        self.identifier = identifier
        
class SemanticGroup(ASTNode):
    def __init__(self, identifier, prompt, k, alias, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identifier = identifier
        self.prompt = prompt
        self.k = k
        self.alias = alias
        
class SemanticFilter(ASTNode):
    def __init__(self, identifier, prompt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identifier = identifier
        self.prompt = prompt
