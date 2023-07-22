from enum import Enum

from .parser_ import SQLParser
from .lexer import SQLLexer
from .ast import select, delete, insert, update, create, explain, alter

andb_lexer = SQLLexer()
andb_parser = SQLParser()


class CmdType(Enum):
    CMD_INSERT = 0
    CMD_DELETE = 1
    CMD_SELECT = 2
    CMD_UPDATE = 3
    CMD_UTILITY = 4
    CMD_UNDEFINED = 5


def andb_query_parse(query):
    return andb_parser.parse(andb_lexer.tokenize(query))


def get_ast_type(ast_):
    if isinstance(ast_, select.Select):
        return CmdType.CMD_SELECT
    elif isinstance(ast_, insert.Insert):
        return CmdType.CMD_INSERT
    elif isinstance(ast_, update.Update):
        return CmdType.CMD_UPDATE
    elif isinstance(ast_, delete.Delete):
        return CmdType.CMD_DELETE
    elif (
            isinstance(ast_, create.CreateTable) or
            isinstance(ast_, create.CreateTable) or
            isinstance(ast_, explain.Explain) or
            isinstance(ast_, alter.AlterTable)
    ):
        return CmdType.CMD_UTILITY
    else:
        return CmdType.CMD_UNDEFINED
