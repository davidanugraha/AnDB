import os
from andb.catalog.oid import INVALID_OID, OID_SCANNING_FILE
from andb.catalog.syscache import CATALOG_ANDB_ATTRIBUTE, CATALOG_ANDB_CLASS, CATALOG_ANDB_TYPE
from andb.errno.errors import AnDBNotImplementedError, InitializationStageError
from andb.executor.operator.logical import *
from andb.executor.operator.physical.utility import CreateIndexOperator, CreateMemoryTableOperator, CreateTableOperator, ExplainOperator, DropTableOperator, DropMemoryTableOperator, DropIndexOperator, CommandOperator
from andb.runtime import session_vars
from andb.sql.parser.ast.create import CreateTable, CreateIndex, CreateMemoryTable, DropMemoryTable
from andb.sql.parser.ast.delete import Delete
from andb.sql.parser.ast.drop import DropIndex, DropTable
from andb.sql.parser.ast.explain import Explain
from andb.sql.parser.ast.insert import Insert
from andb.sql.parser.ast.join import Join
from andb.sql.parser.ast.misc import Star
from andb.sql.parser.ast.operation import Function
from andb.sql.parser.ast.select import Select
from andb.sql.parser.ast.semantic import Prompt, FileSource, SemanticTabular, SemanticGroup, SemanticMatch
from andb.sql.parser.ast.update import Update
from andb.storage.engines.heap.relation import RelationKinds
from andb.sql.parser.ast.utility import Command
from .base import BaseTransformation

from ...executor.operator.utils import expression_eval


class UtilityTransformation(BaseTransformation):
    @staticmethod
    def match(ast) -> bool:
        return isinstance(ast, (CreateIndex, CreateTable, DropTable, DropIndex, Explain, Command))

    @staticmethod
    def on_transform(ast):
        physical_operator = None
        if isinstance(ast, CreateIndex):
            fields = [id_.parts for id_ in ast.columns]
            physical_operator = CreateIndexOperator(index_name=ast.name.parts, table_name=ast.table_name.parts,
                                                    fields=fields, database_oid=session_vars.SessionVars.database_oid,
                                                    index_type=ast.index_type)
        elif isinstance(ast, CreateTable):
            if isinstance(ast, CreateMemoryTable):
                physical_operator = CreateMemoryTableOperator(
                    table_name=ast.name.parts, fields=ast.columns, 
                    database_oid=session_vars.SessionVars.database_oid,
                    temporary=ast.temporary
                )
            else:
                physical_operator = CreateTableOperator(
                    table_name=ast.name.parts, fields=ast.columns, database_oid=session_vars.SessionVars.database_oid
                )
        elif isinstance(ast, DropTable):
            if isinstance(ast, DropMemoryTable):
                physical_operator = DropMemoryTableOperator(
                    table_name=ast.name.parts, database_oid=session_vars.SessionVars.database_oid
                )
            else:
                physical_operator = DropTableOperator(
                    table_name=ast.name.parts, database_oid=session_vars.SessionVars.database_oid
                )
        elif isinstance(ast, DropIndex):
            physical_operator = DropIndexOperator(
                index_name=ast.name.parts, database_oid=session_vars.SessionVars.database_oid
            )
        elif isinstance(ast, Explain):
            physical_operator = ExplainOperator(logical_plan=andb_ast_transform(ast.target))
        elif isinstance(ast, Command):
            physical_operator = CommandOperator(ast.command, ast.parameters)

        return UtilityOperator(physical_operator)


class ConditionTransformation(BaseTransformation):
    @staticmethod
    def match(ast) -> bool:
        return isinstance(ast, Condition)

    @staticmethod
    def on_transform(ast: Condition):
        def swap(node: Condition):
            # let column is at left hand side
            if isinstance(node.right, AbstractColumn):
                node.left, node.right = node.right, node.left
            return node

        def dfs(node: Condition):
            if node is None:
                return
            if not isinstance(node, Condition):
                return node
            left_node = dfs(node.left)
            right_node = dfs(node.right)

            # only convert constant
            if (not isinstance(left_node, AbstractColumn) and
                    not isinstance(right_node, AbstractColumn)):
                return expression_eval(node.expr.value, node.left, node.right)
            else:
                node.left = left_node
                node.right = right_node
                swap(node)
            return node

        return dfs(ast)


class QueryLogicalPlanTransformation(BaseTransformation):
    @staticmethod
    def match(query) -> bool:
        return isinstance(query, LogicalQuery) and len(query.children) == 0

    @staticmethod
    def process_non_join_scan(query: LogicalQuery):
        assert len(query.scan_operators) == 1  #TODO: support 0 in future
        scan = query.scan_operators[0]

        if not query.condition:
            query.add_child(scan)
            return

        # simple predicate pushdown
        scan.condition = query.condition
        scan.table_columns = []
        for table_column in query.get_seen_table_columns(scan.table_name):
            #TODO: because there may be function
            scan.table_columns.append(table_column)
        # column prune:
        # because this is non-join query, it is very simple. if this query has join clause, we have to
        # add join column for the scan operator.
        query.add_child(scan)

    @staticmethod
    def process_join_scan(query: LogicalQuery):
        condition_table_names = {}
        if query.condition and not isinstance(query.condition, SemanticCondition):
            for condition in query.condition.get_iterator():
                if isinstance(condition.left, TableColumn):
                    condition_table_names[condition.left.table_name] = condition
                if isinstance(condition.right, TableColumn):
                    condition_table_names[condition.right.table_name] = condition
        elif query.condition and isinstance(query.condition, SemanticCondition):
            for table_col in query.condition.table_columns:
                if table_col.table_name not in condition_table_names:
                    condition_table_names[table_col.table_name] = query.condition

        scan_operator: "ScanOperator"
        for scan_operator in query.scan_operators:
            # if the query has a condition and this condition only contains one table,
            # push down the predicate (condition).
            if query.condition and len(condition_table_names) == 1 \
                    and scan_operator.table_name in condition_table_names:
                scan_operator.condition = query.condition

            scan_operator.table_columns = []
            for table_column in query.get_seen_table_columns(scan_operator.table_name):
                #TODO: because there may be function
                scan_operator.table_columns.append(table_column)

        # add table columns that come from join conditions
        join_table_columns = []
        for join_operator in query.join_operators:   
            if isinstance(join_operator, SemanticJoinOperator):
                join_table_columns.extend(join_operator.condition.table_columns)
            elif not join_operator.join_condition:
                # skip cross join
                continue
            else:
                for condition in join_operator.join_condition.get_iterator():
                    if isinstance(condition.left, TableColumn):
                        join_table_columns.append(condition.left)
                    if isinstance(condition.right, TableColumn):
                        join_table_columns.append(condition.right)
                #TODO: can be further pruned
                # join_operator.table_columns = None
        
        for join_table_column in join_table_columns:
            for scan_operator in query.scan_operators:
                if scan_operator.table_name != join_table_column.table_name:
                    continue
                if join_table_column not in scan_operator.table_columns:
                    scan_operator.table_columns.append(join_table_column)

        if len(set(condition_table_names.values())) > 1 or len(query.join_operators) > 1:
            raise NotImplementedError('not supported multiple tables join')

        query.add_child(query.join_operators[0])

    @staticmethod
    def process_groupby(query: LogicalQuery):
        aggregation_functions = []
        for target in query.target_list:
            if isinstance(target, FunctionColumn) and hasattr(AggregationFunctions, target.function_name):
                aggregation_functions.append(target)
        if len(aggregation_functions) == 0:
            return
        if len(aggregation_functions) > 1:
            raise NotImplementedError('not support one more aggregations.')
        
        # If query's groupby columns contain semantic groupby, add a semantictransform as child operator
        semantic_groupby_columns = []
        reg_groupby_columns = []
        for gb_col in query.groupby_columns:
            if isinstance(gb_col, SemanticTransformColumn):
                semantic_groupby_columns.append(gb_col)
                target_table_name = None
                for attr_form in query.table_attr_forms[gb_col.table_name]:
                    if attr_form.name == gb_col.column_name:
                        target_table_name = gb_col.table_name
                        break
                if target_table_name is None:
                    reg_groupby_columns.append(VirtualColumn(gb_col.column_name))
                else:
                    reg_groupby_columns.append(TableColumn(target_table_name, gb_col.column_name))
            else:
                reg_groupby_columns.append(gb_col)
        if len(semantic_groupby_columns) != 0:
            operator = SemanticTransformOperator(semantic_groupby_columns, query.children)
            query.children = [operator]
        
        aggregation_function = aggregation_functions[0]
        groupby_columns = set(reg_groupby_columns)
        groupby_columns.add(aggregation_function)
        if groupby_columns != set(query.target_list):
            raise InitializationStageError('not found all columns are in the group by list.')
        operator = GroupOperator(group_by_columns=reg_groupby_columns,
                                 aggregate_function=aggregation_function,
                                 having_clause=query.having_clause)
        # set the group by operator onto all nodes.
        operator.children = query.children
        query.children = [operator]

    @staticmethod
    def on_transform(query: LogicalQuery):
        #TODO: extract all involved columns, then prune useless columns
        #TODO: rewrite

        if not query.join_operators:
            QueryLogicalPlanTransformation.process_non_join_scan(query)
        else:
            QueryLogicalPlanTransformation.process_join_scan(query)

        #TODO: limit, ...
        if query.sort_clause:
            query.sort_clause.children = query.children
            query.children = [query.sort_clause]

        # process function
        for target in query.target_list:
            if isinstance(target, FunctionColumn):
                if hasattr(AggregationFunctions, target.function_name):
                    pass
                else:
                    raise NotImplementedError('not supported normal functions.')

        QueryLogicalPlanTransformation.process_groupby(query)

        return query


class SelectTransformation(BaseTransformation):
    @staticmethod
    def match(ast) -> bool:
        return isinstance(ast, Select)

    @staticmethod
    def _supplement_table_name_for_column(column: TableColumn, table_attr_forms):
        if column.table_name is not None:
            found = False
            for attr_form in table_attr_forms[column.table_name]:
                if attr_form.name == column.column_name:
                    found = True
                    break
            if not found:
                raise InitializationStageError(f"not found '{str(column)}'.")
        else:
            column_table_name = None
            for table_name in table_attr_forms:
                for attr_form in table_attr_forms[table_name]:
                    if attr_form.name == column.column_name:
                        if column_table_name is not None:
                            raise InitializationStageError(
                                f'both table {column_table_name} and {table_name} have'
                                f' the same column {column.column_name}.')
                        column_table_name = table_name
            if column_table_name is None:
                raise InitializationStageError(f"not found '{column.column_name}'.")
            column.table_name = column_table_name
                        
    @staticmethod
    def _supplement_table_name(condition: Condition, table_attr_forms):
        for node in condition.get_iterator():
            for arg in (node.left, node.right):
                if isinstance(arg, TableColumn):
                    SelectTransformation._supplement_table_name_for_column(arg, table_attr_forms)
                elif isinstance(arg, FunctionColumn):
                    for column in arg.columns:
                        if isinstance(column, TableColumn):
                            SelectTransformation._supplement_table_name_for_column(column, table_attr_forms)
        return condition

    @classmethod
    def _find_table_name(cls, query, lookup_column_name):
        target_table_name = None
        for table_name in query.from_tables:
            for attr_form in query.table_attr_forms[table_name]:
                if attr_form.name == lookup_column_name:
                    if target_table_name is not None:
                        raise InitializationStageError(f'both table {target_table_name} and {table_name} have'
                                                       f' the same column {lookup_column_name}.')
                    target_table_name = table_name
        return target_table_name
    
    @classmethod
    def _get_semantic_condition(cls, sem_match, query):
        # Get the table columns by converting identifiers into TableColumn
        table_columns = []
        for target in sem_match.identifiers:
            table_name, column_name = None, None
            if isinstance(target, Identifier) and '.' in target.parts:
                # Get table and column name based on DOT
                items = target.parts.split('.')
                if len(items) != 2:
                    raise InitializationStageError(f"syntax error: '{target.parts}'.")
                table_name, column_name = items
                if table_name not in query.from_tables:
                    raise InitializationStageError(f"not found '{table_name}'.")
                found_column_name = False
                for attr_form in query.table_attr_forms[table_name]:
                    if attr_form.name == column_name:
                        found_column_name = True
                        break
                if not found_column_name:
                    raise InitializationStageError(f"not found column '{column_name}' in '{table_name}'.")
            elif isinstance(target, Identifier):
                # Get table name based on column name
                column_name = target.parts
                table_name = cls._find_table_name(query, column_name)
                if table_name is None:
                    raise InitializationStageError(f"not found '{table_name}'.")
            else:
                raise InitializationStageError(f"Target is not Identifier.")
            table_columns.append(TableColumn(table_name, column_name))
            
        return SemanticCondition(sem_match.condition, table_columns)

    @classmethod
    def transform_from_clause(cls, ast_outer, query):
        dont_add_table_as_scan = []
        def inner_transform(ast):
            unchecked_tables = []
            unchecked_files = []
            if isinstance(ast, Identifier):
                unchecked_tables.append(ast.parts)
            elif isinstance(ast, Join):
                inner_transform(ast.left)
                inner_transform(ast.right)
            elif isinstance(ast, FileSource):
                unchecked_files.append(ast.file_path.value)
            elif isinstance(ast, SemanticTabular):
                # Process the table source within the SemanticTabular
                inner_transform(ast.identifier)
                inner_transform(ast.table_source)
                
                semantic_schemas = ast.semantic_schemas
                table_name = ast.identifier.parts
                prompt_columns = []
                for prompt_col, attr_form in zip(semantic_schemas, query.table_attr_forms[table_name]):
                    if not isinstance(prompt_col, Prompt):
                        raise InitializationStageError(f"Invalid schema definition: {prompt_col}")

                    column = PromptColumn(
                        table_name=table_name,
                        column_name=attr_form.name,
                        prompt_text=prompt_col.prompt_text
                    )
                    prompt_columns.append(column)
                
                # Don't add temp table or source table as scan, it's already created below
                table_source_scan = None
                if isinstance(ast.table_source, FileSource):
                    # Currently only support for file
                    file_path = ast.table_source.file_path.value
                    table_source_scan = ScanOperator(file_path, table_oid=query.from_tables[file_path])
                    dont_add_table_as_scan.append(file_path)
                else:
                    raise InitializationStageError(f"Source not file is not supported currently...")
                dont_add_table_as_scan.append(ast.identifier.parts)
                
                if table_source_scan is not None:
                    table_source_scan = [table_source_scan]

                semantic_op = SemanticScanOperator(
                    table_name=table_name,
                    prompt_columns=prompt_columns,
                    condition=None,
                    children=table_source_scan
                )
                query.scan_operators.append(semantic_op)
            else:
                raise NotImplementedError()

            for table_name in unchecked_tables:
                table_oid = CATALOG_ANDB_CLASS.get_relation_oid(table_name, database_oid=session_vars.SessionVars.database_oid,
                                                                kind=None)
                if table_oid != INVALID_OID:
                    query.from_tables[table_name] = table_oid
                    query.table_attr_forms[table_name] = CATALOG_ANDB_ATTRIBUTE.get_table_forms(table_oid)
                else:
                    raise InitializationStageError(f'not found the table {table_name}.')

            for file_path in unchecked_files:
                # scan the file from current working director
                real_file_path = os.path.join(os.path.realpath(f'./base/{session_vars.SessionVars.database_oid}/files'), file_path)
                if os.path.exists(real_file_path):
                    query.from_tables[file_path] = OID_SCANNING_FILE
                    query.table_attr_forms[file_path] = CATALOG_ANDB_ATTRIBUTE.get_table_forms(OID_SCANNING_FILE)
                else:
                    raise InitializationStageError(f'not found the file {file_path} from files directory.')

        inner_transform(ast_outer)
        # scan operator
        for name_or_path in query.from_tables:
            # Don't add TempTable by SemanticTabular as ScanOperator
            if name_or_path not in dont_add_table_as_scan:
                query.scan_operators.append(ScanOperator(name_or_path, table_oid=query.from_tables[name_or_path]))

    @classmethod
    def transform_target_list(cls, ast, query):
        for target in ast:
            # parse star
            if isinstance(target, Star):
                for table_name in query.from_tables:
                    if query.from_tables[table_name] != OID_SCANNING_FILE:
                        # Skip file's attributes, which is content and embedding
                        for attr_form in query.table_attr_forms[table_name]:
                            column = TableColumn(table_name, attr_form.name)
                            if target.alias:
                                column.alias = target.alias.parts
                            query.target_list.append(column)
                            query.add_seen_table_column(column)
            elif isinstance(target, Identifier) and '.' in target.parts:
                items = target.parts.split('.')
                if len(items) != 2:
                    raise InitializationStageError(f"syntax error: '{target.parts}'.")
                table_name, column_name = items
                if table_name not in query.from_tables:
                    raise InitializationStageError(f"not found '{target.parts}'.")
                found = False
                for attr_form in query.table_attr_forms[table_name]:
                    if attr_form.name == column_name:
                        found = True
                        column = TableColumn(table_name, column_name)
                        break
                if not found:
                    raise InitializationStageError(f"not found '{target.parts}'.")
                if target.alias:
                    column.alias = target.alias.parts
                query.target_list.append(column)
                query.add_seen_table_column(column)
            elif isinstance(target, Identifier):
                target_column_name = target.parts
                target_table_name = cls._find_table_name(query, target_column_name)
                if target_table_name is None:
                    # TODO: What if there are multiple target_table_name, then it's invalid?
                    query.unchecked_columns.append(target_column_name)
                    continue
                found = False
                for attr_form in query.table_attr_forms[target_table_name]:
                    if attr_form.name == target_column_name:
                        found = True
                        column = TableColumn(target_table_name, target_column_name)
                        break
                if not found:
                    raise InitializationStageError(f"not found '{target.parts}'.")
                else:
                    if target.alias:
                        column.alias = target.alias.parts
                    query.target_list.append(column)
                    query.add_seen_table_column(column)
            elif isinstance(target, Function):
                #TODO: multiple parameters
                table_columns = []
                for id_ in target.args:
                    target_column_name = id_.parts
                    target_table_name = cls._find_table_name(query, target_column_name)
                    table_column = TableColumn(target_table_name, target_column_name)
                    table_column.function_name = target.op
                    table_columns.append(table_column)
                    query.add_seen_table_column(table_column)
                function_column = FunctionColumn(target.op, table_columns)
                if target.alias:
                    function_column.alias = target.alias.parts
                query.target_list.append(function_column)
            elif isinstance(target, Prompt):
                prompt_column = PromptColumn(target.prompt_text)
                if target.alias:
                    prompt_column.alias = target.alias.parts
                query.target_list.append(prompt_column)
            else:
                #TODO: function and agg
                raise NotImplementedError('not supported this syntax.')

    @classmethod
    def transform_where_clause(cls, ast, query):
        if ast is not None:
            if isinstance(ast, SemanticMatch):
                query.condition = cls._get_semantic_condition(ast, query)
            else:        
                where_condition = ConditionTransformation.on_transform(Condition(ast))
                if isinstance(where_condition, bool):
                    if where_condition:
                        query.condition = None  # we don't need condition
                    else:
                        raise NotImplementedError('should return empty set directly.')
                elif isinstance(where_condition, Condition):
                    # supplement missing table name and check existing table name
                    query.condition = cls._supplement_table_name(where_condition, query.table_attr_forms)
                else:
                    raise NotImplementedError('not supported this syntax.')

    @classmethod
    def transform_join_clause(cls, ast, query):
        if isinstance(ast, Join):
            # maybe it is a multi-way join
            if isinstance(ast.left, Join):
                SelectTransformation.transform_join_clause(ast.left, query)
            # right should always be a table for now
            if isinstance(ast.right, Join):
                SelectTransformation.transform_join_clause(ast.right, query)

            join_clause = ast
            
            # Get left and right table names
            if isinstance(join_clause.left, SemanticTabular):
                left_table_name = join_clause.left.identifier.parts
            else:
                left_table_name = join_clause.left.parts
            if isinstance(join_clause.right, SemanticTabular):
                right_table_name = join_clause.right.identifier.parts
            else:
                right_table_name = join_clause.right.parts
            
            if not join_clause.implicit:
                if isinstance(join_clause.condition, SemanticMatch):
                    semantic_condition = cls._get_semantic_condition(join_clause.condition, query)
                    join_operator = SemanticJoinOperator(condition=semantic_condition,
                                                         children_table_names=[left_table_name, right_table_name],
                                                         join_type=join_clause.join_type)
                else:
                    join_condition = ConditionTransformation.on_transform(Condition(join_clause.condition))
                    join_condition = cls._supplement_table_name(join_condition, query.table_attr_forms)
                    join_operator = JoinOperator(join_condition=join_condition,
                                            join_type=join_clause.join_type)
            else:
                join_operator = JoinOperator(join_condition=None,
                                            join_type=join_clause.join_type)

            left_scan_operator = right_scan_operator = None
            for scan_operator in query.scan_operators:
                # for self-joining, the left and right table reuse a same scan operator
                if scan_operator.table_name == left_table_name:
                    left_scan_operator = scan_operator
                if scan_operator.table_name == right_table_name:
                    right_scan_operator = scan_operator

            join_operator.add_child(left_scan_operator)
            join_operator.add_child(right_scan_operator)

            query.join_operators.append(join_operator)

    @classmethod
    def transform_order_clause(cls, ast, query):
        if ast:
            sort_columns = []
            ascending_orders = []
            for node in ast:
                table_column = TableColumn(table_name=DummyTableName.UNKNOWN, column_name=None)
                for table_name in query.table_attr_forms:
                    for attr_form in query.table_attr_forms[table_name]:
                        if attr_form.name == node.attr.parts:
                            if table_column.column_name is not None:
                                raise InitializationStageError(
                                    f'found duplicated column name {table_column.column_name} '
                                    f'from {table_column.table_name} and {table_name}.'
                                )
                            else:
                                table_column.table_name = table_name
                                table_column.column_name = attr_form.name
                                break
                sort_columns.append(table_column)
                ascending_orders.append(node.direction == 'ASC')

            query.sort_clause = SortOperator(sort_columns, ascending_orders)

    @classmethod
    def transform_group_clause(cls, ast, query):
        if ast.group_by:
            for id_ in ast.group_by:
                if isinstance(id_, SemanticGroup):
                    # Find the column that this GroupBy is referring to, and then add as GroupByColumn
                    column_alias = id_.alias.parts
                    original_column = id_.identifier.parts # TODO: Support multiple identifiers
                    table_name = cls._find_table_name(query, original_column)
                    if table_name is None:
                        raise InitializationStageError(f"not found '{original_column}'.")
                    if not isinstance(id_.prompt, Prompt):
                        raise InitializationStageError(f"SemanticGroupBy is not provided with prompt.")
                    semantic_column = SemanticTransformColumn(table_name, [original_column], column_alias,
                                                              id_.prompt.prompt_text, id_.k)
                    query.groupby_columns.append(semantic_column)
                    
                    if column_alias in query.unchecked_columns:
                        table_column = VirtualColumn(column_alias)
                        query.unchecked_columns.remove(column_alias)
                        query.target_list.append(table_column)
                    else:
                        table_column = TableColumn(table_name, column_alias)
                        
                    query.add_seen_table_column(table_column)
                else:
                    column_name = id_.parts
                    table_name = cls._find_table_name(query, column_name)
                    table_column = TableColumn(table_name, column_name)
                    query.groupby_columns.append(table_column)
                    query.add_seen_table_column(table_column)

        if ast.having:
            if not ast.group_by:
                raise NotImplementedError('only support having clause for group by clause.')
            having_clause = Condition(ast.having)
            cls._supplement_table_name(having_clause, query.table_attr_forms)
            query.having_clause = ConditionTransformation.on_transform(having_clause)

    @classmethod
    def on_transform(cls, ast: Select):
        query = LogicalQuery()

        cls.transform_from_clause(ast.from_table, query)
        cls.transform_target_list(ast.targets, query)
        cls.transform_where_clause(ast.where, query)
        cls.transform_join_clause(ast.from_table, query)
        cls.transform_order_clause(ast.order_by, query)
        cls.transform_group_clause(ast, query)  # we need to pass both ast.group_by and ast.having
        
        # Unchecked column still exists
        if len(query.unchecked_columns) > 0:
            raise InitializationStageError(f"not found '{query.unchecked_columns[0]}'.")

        #TODO: distinct
        #TODO: limit

        query.distinct = ast.distinct

        if QueryLogicalPlanTransformation.match(query):
            query = QueryLogicalPlanTransformation.on_transform(query)

        return query


class InsertTransformation(BaseTransformation):
    @staticmethod
    def match(ast) -> bool:
        return isinstance(ast, Insert)

    @staticmethod
    def on_transform(ast: Insert):
        table_oid = CATALOG_ANDB_CLASS.get_relation_oid(relation_name=ast.table.parts,
                                                        database_oid=session_vars.SessionVars.database_oid)
        if table_oid == INVALID_OID:
            raise InitializationStageError(f'cannot get oid for the table {ast.table.parts}.')

        attr_forms = CATALOG_ANDB_ATTRIBUTE.search(lambda r: r.class_oid == table_oid)
        if not attr_forms:
            raise InitializationStageError(f'cannot get the table {ast.table.parts}.')

        rows = []
        for value in ast.values:
            row = [None for _ in range(len(attr_forms))]
            if isinstance(value, Constant):
                row[0] = CATALOG_ANDB_TYPE.get_type_form_by_oid(
                    attr_forms[0].type_oid).format_value(value.value)
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    attr_num = attr_forms[i].num
                    row[attr_num] = CATALOG_ANDB_TYPE.get_type_form_by_oid(
                        attr_forms[i].type_oid).format_value(v.value)
            else:
                raise NotImplementedError('not supported this syntax yet.')
            rows.append(row)
            for i in range(len(attr_forms)):
                if row[i] is None and attr_forms[i].notnull:
                    raise InitializationStageError(f'{attr_forms[i].name} should not be null.')

        if ast.columns:
            columns = [id_.parts for id_ in ast.columns]
        else:
            columns = [attr.name for attr in attr_forms]

        operator = InsertOperator(
            table_name=ast.table.parts, table_oid=table_oid, columns=columns,
            values=rows, select=None
        )
        if ast.from_select:
            select_logical_plan = SelectTransformation.on_transform(ast.from_select)
            operator.select = select_logical_plan

        return operator


class DeleteTransformation(BaseTransformation):
    @staticmethod
    def match(ast) -> bool:
        return isinstance(ast, Delete)

    @staticmethod
    def on_transform(ast: Delete):
        # transform to a query
        #TODO: extract where predicates
        select = Select(targets=[Star()])
        select.from_table = ast.table
        select.where = ast.where
        query = SelectTransformation.on_transform(select)
        return DeleteOperator(ast.table.parts, query)

class UpdateTransformation(BaseTransformation):
    @staticmethod
    def match(ast) -> bool:
        return isinstance(ast, Update)

    @staticmethod
    def on_transform(ast: Update):
        table_oid = CATALOG_ANDB_CLASS.get_relation_oid(relation_name=ast.table.parts,
                                                        database_oid=session_vars.SessionVars.database_oid)
        if table_oid == INVALID_OID:
            raise InitializationStageError(f'cannot get oid for the table {ast.table.parts}.')
    
        columns = []
        values = []
        for column_name, value_expr in ast.columns.items():
            columns.append(TableColumn(table_name=ast.table.parts,
                                       column_name=column_name))
            # format the value to avoid type mismatch
            attr_form = CATALOG_ANDB_ATTRIBUTE.get_table_attr(table_oid, column_name)
            type_form = CATALOG_ANDB_TYPE.get_type_form_by_oid(attr_form.type_oid)
            if isinstance(value_expr, Constant):
                values.append(type_form.format_value(value_expr.value))
            else:
                raise NotImplementedError('not supported this syntax yet.')
        condition = ConditionTransformation.on_transform(Condition(ast.where))

        select = Select(targets=[Star()])
        select.from_table = ast.table
        select.where = ast.where
        query = SelectTransformation.on_transform(select)

        operator = UpdateOperator(table_name=ast.table.parts, query=query, columns=columns,
                                  values=values, condition=condition)
        return operator


class SemanticQueryTransformation(BaseTransformation):
    pass


_all_transformations = [trans() for trans in BaseTransformation.__subclasses__()]


def andb_ast_transform(ast):
    for trans in _all_transformations:
        if trans.match(ast):
            return trans.on_transform(ast)
    raise AnDBNotImplementedError('not supported this grammar yet')
