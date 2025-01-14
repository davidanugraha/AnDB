from andb.sql.parser import andb_query_parse
from andb.sql.parser.ast.explain import Explain
from andb.sql.parser.ast.semantic import Prompt, FileSource, SemanticTabular
from andb.sql.parser.ast.join import Join, JoinType

SETUP = "SETUP"
MAIN_QUERY = "MAIN_QUERY"
CLEANUP = "CLEANUP"

class Stage:
    def __init__(self, stage_type, ast, cleanup_ast=None):
        self.stage_type = stage_type
        self.ast = ast
        self.cleanup_ast = cleanup_ast  # Optional cleanup AST
        self._executed = False
        
    def is_success(self):
        return self._executed

    def mark_success(self):
        self._executed = True
        
    def get_ast(self):
        return self.ast
    
    def get_cleanup_ast(self):
        return self.cleanup_ast
        
    def has_output(self):
        return self.stage_type == MAIN_QUERY

def _create_temp_table_stage(semantic_tabular):
    # Modify the main query AST to reference the temporary table
    temp_table_name = semantic_tabular.identifier.parts
    columns_definition = []
    for prompt_col in semantic_tabular.semantic_schemas:
        assert(isinstance(prompt_col, Prompt) and len(prompt_col.defined_column) == 2)
        field_name, field_type = prompt_col.defined_column
        columns_definition.append(f"{field_name} {field_type}")
    columns_definition_str = ", ".join(columns_definition)
        
    create_temp_table_query = f"CREATE TEMPORARY TABLE {temp_table_name} ({columns_definition_str})" # TODO: Is this dangerous?
    create_table_ast = andb_query_parse(create_temp_table_query)
    
    drop_temp_table_query = f"DROP TEMPORARY TABLE {temp_table_name}" # TODO: Is this dangerous?
    drop_table_ast = andb_query_parse(drop_temp_table_query)

    return Stage(stage_type=SETUP, ast=create_table_ast, cleanup_ast=drop_table_ast)

def _create_main_query_stage(ast):
    return Stage(stage_type=MAIN_QUERY, ast=ast)

def _create_join_stages(semantic_join:Join, list_stages:list):
    if isinstance(semantic_join.left, Join):
        _create_join_stages(semantic_join.left, list_stages)
    elif isinstance(semantic_join.left, SemanticTabular):
        list_stages.append(_create_temp_table_stage(semantic_join.left))
    
    if isinstance(semantic_join.right, SemanticTabular):
        list_stages.append(_create_temp_table_stage(semantic_join.right))

def andb_decompose_ast(original_ast):
    """Decompose the original AST into multiple stages.""" # TODO: Generalize
    list_stages = []

    if isinstance(original_ast, Explain):
        curr_ast = original_ast.target
    else:
        curr_ast = original_ast

    if hasattr(curr_ast, "from_table"):
        if isinstance(curr_ast.from_table, SemanticTabular):
            list_stages.append(_create_temp_table_stage(curr_ast.from_table))
        elif isinstance(curr_ast.from_table, Join):
            _create_join_stages(curr_ast.from_table, list_stages)

    list_stages.append(_create_main_query_stage(original_ast))
        
    return list_stages
