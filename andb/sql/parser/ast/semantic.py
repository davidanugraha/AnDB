from .base import ASTNode

class FileSource(ASTNode):
    def __init__(self, file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.parts = file_path.value

class Prompt(ASTNode):
    def __init__(self, prompt_text, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_text = prompt_text
        self.alias = None
        
class SemanticSchemas(ASTNode):
    def __init__(self, schema_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schema_list = schema_list
        
class SemanticTabular(ASTNode):
    def __init__(self, semantic_schemas, table_source, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_schemas = semantic_schemas
        self.table_source = table_source
        self.alias = None 
