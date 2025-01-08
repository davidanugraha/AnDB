from .stager import andb_decompose_ast
from .transformations import andb_ast_transform
from .implementations import andb_logical_plan_implement

def andb_get_stages(ast):
    return andb_decompose_ast(ast)

def andb_query_plan(ast):
    logical_plan = andb_ast_transform(ast)
    physical_plan = andb_logical_plan_implement(logical_plan)
    return physical_plan
