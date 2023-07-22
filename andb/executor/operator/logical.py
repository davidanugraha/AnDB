from enum import Enum

from andb.sql.parser.ast.operation import BinaryOperation
from andb.sql.parser.ast.misc import Constant
from andb.sql.parser.ast.identifier import Identifier


class LogicalOperator:
    OPERATOR_NAME = 'name'
    OPERATOR_CHILDREN = 'children'

    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []

    def add_child(self, child_operator):
        self.children.append(child_operator)

    def get_args(self):
        return {}

    def to_dict(self):
        logical_plan = {
            self.OPERATOR_NAME: self.name,
            self.OPERATOR_CHILDREN: []
        }
        logical_plan.update(self.get_args())

        for child_operator in self.children:
            child_logical_plan = child_operator.to_dict()
            logical_plan['children'].append(child_logical_plan)
        return logical_plan


class ExprOperation(Enum):
    PLUS = '+'
    MINUS = '-'
    DIVIDE = '/'
    MODULO = '%'
    EQ = '='
    NE = '!='
    GEQ = '>='
    GT = '>'
    LEQ = '<='
    LT = '<'
    AND = 'and'
    OR = 'or'
    IS_NOT = 'is not'
    NOT = 'not'
    IS = 'is'
    LIKE = 'like'
    IN = 'in'


class TableColumn:
    def __init__(self, table_name, column_name):
        self.table_name = table_name
        assert column_name
        self.column_name = column_name

    def __repr__(self):
        return f'{self.table_name}.{self.column_name}'

    def need_to_fill(self):
        return self.table_name is None


class Condition(LogicalOperator):
    def __init__(self, operation, children=None):
        super().__init__('Expression', children)
        assert isinstance(operation, BinaryOperation)
        self.expr = None
        for o in ExprOperation:
            if o.value == operation.op:
                self.expr = o
                break
        assert self.expr

        self.left = self._convert(operation.args[0])
        self.right = self._convert(operation.args[1])

    @staticmethod
    def _convert(item):
        if isinstance(item, Constant):
            return item.value
        elif isinstance(item, Identifier):
            return TableColumn(table_name=None, column_name=item.parts)
        else:
            raise

    def add_child(self, child_operator):
        assert isinstance(child_operator, Condition)
        super().add_child(child_operator)

    def get_args(self):
        return {'expression': f'({self.left} {self.expr.value} {self.right})'}


class ProjectionOperator(LogicalOperator):
    def __init__(self, columns, children=None):
        super().__init__('Projection', children)
        self.columns = columns

    def get_args(self):
        return {'columns': self.columns}


class SelectionOperator(LogicalOperator):
    def __init__(self, condition: Condition, children=None):
        super().__init__('Selection', children)
        self.condition = condition

    def get_args(self):
        return {'condition': self.condition}


class JoinOperator(LogicalOperator):
    def __init__(self, join_condition: Condition, children=None):
        super().__init__('Join', children)
        self.join_condition = join_condition

    def get_args(self):
        return {'join_condition': self.join_condition}


class GroupOperator(LogicalOperator):
    def __init__(self, group_by_columns, aggregate_functions, children=None):
        super().__init__('Group', children)
        self.group_by_columns = group_by_columns
        self.aggregate_functions = aggregate_functions

    def get_args(self):
        return {
            'group_by_columns': self.group_by_columns,
            'aggregate_functions': self.aggregate_functions
        }


class ScanOperator(LogicalOperator):
    def __init__(self, table_name, condition: Condition = None):
        super().__init__('Scan')
        self.table_name = table_name
        self.condition = condition

    def get_args(self):
        return {'table_name': self.table_name,
                'condition': self.condition}


class SortOperator(LogicalOperator):
    def __init__(self, sort_columns, children=None):
        super().__init__('Sort', children)
        self.sort_columns = sort_columns

    def get_args(self):
        return {'sort_columns': self.sort_columns}


class DuplicateRemovalOperator(LogicalOperator):
    def __init__(self, children=None):
        super().__init__('DuplicateRemoval', children)


class LimitOperator(LogicalOperator):
    def __init__(self, limit_count, children=None):
        super().__init__('Limit', children)
        self.limit_count = limit_count

    def get_args(self):
        return {'limit_count': self.limit_count}


class UnionOperator(LogicalOperator):
    def __init__(self, children=None):
        super().__init__('Union', children)


class IntersectOperator(LogicalOperator):
    def __init__(self, children=None):
        super().__init__('Intersect', children)


class ExceptOperator(LogicalOperator):
    def __init__(self, children=None):
        super().__init__('Except', children)


class UtilityOperator(LogicalOperator):
    def __init__(self, physical_operator):
        super().__init__('Utility')
        self.physical_operator = physical_operator

    def get_args(self):
        return {'PhysicalOperator': self.physical_operator.__class__.__name__}


class InsertOperator(LogicalOperator):
    def __init__(self, table_name, columns, values=None, select=None):
        super().__init__('Insert')
        self.table_name = table_name
        self.columns = columns
        self.values = values
        self.select = select

    def get_args(self):
        return {'table_name': self.table_name,
                'columns': self.columns,
                'from_select': bool(self.select)}


class DeleteOperator(LogicalOperator):
    def __init__(self, table_name, condition):
        super().__init__('Delete')
        self.table_name = table_name
        self.condition = condition

    def get_args(self):
        return {'table_name': self.table_name,
                'condition': self.condition}


class UpdateOperator(LogicalOperator):
    def __init__(self, table_name, columns, values=None, condition: Condition = None):
        super().__init__('Update')
        self.table_name = table_name
        self.columns = columns
        self.values = values
        self.condition = condition

    def get_args(self):
        return {'table_name': self.table_name,
                'columns': self.columns,
                'condition': self.condition}


def explain_logical_plan(operator: LogicalOperator, indent=''):
    output_lines = []
    name = operator.name
    output_lines.append(f"{indent}├─ {name}")

    for name, value in operator.get_args().items():
        output_lines.append(f"{indent}└  {name}: {value}")

    children_count = len(operator.children)
    for i, child in enumerate(operator.children):
        is_last_child = (i == children_count - 1)
        branch_char = ' ' if is_last_child else '├'
        branch_indent = '    ' if is_last_child else '│   '
        child_indent = f"{indent}{branch_char} {branch_indent}"
        output_lines.extend(explain_logical_plan(child, child_indent))

    return output_lines
