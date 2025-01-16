import json
import logging
import re

from andb.errno.errors import ExecutionStageError
from andb.executor.operator.logical import Condition, DummyTableName, PromptColumn, TableColumn, SemanticTransformColumn, VirtualColumn
from andb.executor.operator.physical.base import PhysicalOperator
from andb.executor.operator.physical.select import Filter
from andb.sql.parser.ast.join import JoinType
from andb.ai.client_model import ClientModelFactory
from andb.ai.embedding_model import EmbeddingModelFactory
from andb.runtime import session_vars


def default_client_model():
    return ClientModelFactory.create_model(model_type=session_vars.SessionParameter.client_llm,
                                           **session_vars.SessionParameter.__dict__)
    
def default_embedding_model():
    return EmbeddingModelFactory.create_model(model_type=session_vars.SessionParameter.embed_llm,
                                              **session_vars.SessionParameter.__dict__)

## HELPER FUNCTIONS

def _parse_json(output):
    try:
        # Try parsing directly first
        return json.loads(output)
    except json.JSONDecodeError:
        # Clean the output for common issues
        cleaned_output = output.strip()

        # Extract potential JSON objects or arrays
        cleaned_entries = []
        json_object_pattern = re.compile(r'\{.*?}', re.DOTALL)
        entries = json_object_pattern.findall(cleaned_output)
        for entry in entries:
            try:
                # Test if each entry is valid JSON
                json.loads(entry)
                cleaned_entries.append(entry)
            except json.JSONDecodeError:
                # Skip invalid entries
                pass

        # Reconstruct the cleaned JSON array
        cleaned_output = "[" + ",".join(cleaned_entries) + "]"

        # Attempt to parse again
        try:
            return json.loads(cleaned_output)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Error cleaning JSON: {e}")

class SemanticPrompt(PhysicalOperator):
    def __init__(self, prompt_text):
        super().__init__('SemanticPrompt')
        self.prompt_text = prompt_text
        self.has_init_models = False
        self.client_model = None
        self.stream = None

    def open(self):
        messages = [{
            "role": "user",
            "content": f"Hello, these are specific requirements: \"{self.prompt_text}\". "
                       f"Then, I am going to give you a text, please generate a response "
                       f"based on the requirements and the text."
        }]
        self.stream = self.client_model.complete_messages(messages=messages, stream=True)

    def next(self):
        if not self.has_init_models:
            self.has_init_models = True
            self.client_model = default_client_model()
        
        for text in self.children[0].next():
            messages = [
                {"role": "assistant", "content": "I understand. Please provide the text."},
                {"role": "user", "content": text}
            ]
            response_stream = self.client_model.complete_messages(messages=messages, stream=True)

            full_response = ""
            try:
                for chunk in response_stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue

            cleaned_response = full_response.strip()
            if cleaned_response:
                yield cleaned_response

    def close(self):
        if self.stream:
            try:
                self.stream.close()
            except:
                pass
        self.stream = None


class SemanticFilter(PhysicalOperator):
    def __init__(self, condition):
        super().__init__('SemanticFilter')
        self.has_init_models = False
        self.client_model = None
        self.condition = condition.condition
        self.filter_table_columns = condition.table_columns
        self.child_columns = []
        self.tab_col_inds = []
        
    def set_proj_index(self, columns):
        # Iterate over which column index to be used; for now assume only left and right
        for filter_col in self.filter_table_columns:
            match_found = False
            for col_index, col in enumerate(columns):
                if filter_col == col: # Compare TableColumn with TableColumn
                    self.tab_col_inds.append(col_index)
                    match_found = True
                    break

            if match_found:
                break
            
            # If no match was found for some reason
            if not match_found:
                raise ValueError(f"Column '{filter_col.column_name}' not found.")

    def open(self):
        if len(self.children) != 1:
            raise ValueError("SemanticFilter currently only supports one input operator")

        self.children[0].open()
        self.columns = self.children[0].columns
        self.set_proj_index(self.columns)
        super().open()
        
    def close(self):
        for child in self.children:
            child.close()
        return super().close()
        
    def _construct_prompt_condition(self, row):
        # Condition is already a string that just needs to be formatted
        entry_vals = []
        for col_ind in self.tab_col_inds:
            entry_vals.append(row[col_ind])
                
        return self.condition.format(*tuple(entry_vals))

    def filter(self, iterator):
        if not self.has_init_models:
            self.has_init_models = True
            self.client_model = default_client_model()
        
        for tuple in iterator:
            temperature = 0.1
            msg = self._construct_prompt_condition(tuple)
            messages = [
                {"role": "system", "content": "You are a strict judge that outputs only 'true' or 'false' with no punctuation or extra characters based on a given statement."},
                {"role": "user", "content": msg}
            ]            
            response = self.client_model.complete_messages(messages, temperature=temperature)
            if response.lower() == "true":
                yield tuple

    def next(self):
        for filtered_tuple in self.filter(self.children[0].next()):
            yield filtered_tuple

class SemanticTransform(PhysicalOperator):
    """Physical operator for processing semantic target list with prompts"""

    def __init__(self, target_columns):
        """
        Args:
            target_columns: List of target columns including prompts
        """
        super().__init__('SemanticTransform')
        self.columns = []
        self.transform_columns = target_columns
        self.need_clustering = False
        for col in self.transform_columns:
            if isinstance(col, SemanticTransformColumn) and (col.k is None or col.k != 1):
                self.need_clustering = True
                break
        
        self.has_init_models = False
        self.client_model = None
        self.embedding_model = None
        self.stream = None
        self.result_tuples = []
        self.projection_indices = []
        self.target_index = []
        self.answer_choice_fld = "answer_choice"

    def open(self):
        """Initialize the operator"""
        if len(self.children) != 1:
            raise ValueError("SemanticTransform requires exactly one input operator")
        self.children[0].open()
        self._get_projection_columns_indices()
    
    def _transform_tuples(self):
        """Simple map/extract using prompting"""
        prompt_system = """You are a helpful assistant that follows the instruction provided by the user."""
        
        for index, col in enumerate(self.transform_columns):
            json_array = self._convert_tuples_to_json(index)

            # Transform
            messages_transform = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": f"""
                Without explanation, Given the following JSON array of objects, extract based on the following instruction: {col.prompt_text}.
                Output only each extraction result in JSON format with field '{col.column_name}'.

                JSON array:
                {str(json_array)}
                """}
            ]
            response = self.client_model.complete_messages(messages_transform, temperature=0.1)
            
            answer_json = _parse_json(response)
            # Check length TODO: Log misalignment?
            if len(answer_json) > len(self.result_tuples):
                # Truncate, but should we put warning?
                answer_json = answer_json[:len(self.result_tuples)]
            elif len(answer_json) < len(self.result_tuples):
                # Unexpected, leaving rest to None, should we put warning?
                answer_json.extend([{col.column_name: None}] * (len(self.result_tuples) - len(answer_json)))  
            
            # Modify answer
            for i, answer in enumerate(answer_json):
                if self.target_index[index] < len(self.result_tuples[i]):
                    self.result_tuples[i][self.target_index[index]] = answer[col.column_name]
                elif self.target_index[index] == len(self.result_tuples[i]):
                    self.result_tuples[i].append(answer[col.column_name])
                else:
                    raise IndexError("Unexpected indexing error when appending answer")
        
    def _generate_mcq_options(self, clustered_categories):
        def index_to_letter(index):
            """Convert an index to an alphabetic label (A, B, ..., AA, AB, ...)."""
            letters = []
            while index >= 0:
                letters.append(chr(65 + (index % 26)))  # Convert to A-Z
                index = index // 26 - 1
            return ''.join(reversed(letters))
        
        # Split the categories into a list
        categories = [cat.strip() for cat in clustered_categories.split(",")]
        
        # Generate Multiple-Choice Question (MCQ) options
        mcq_options = []
        choice_to_category = {}
        for i, category in enumerate(categories):
            choice = index_to_letter(i)  # Generate letter-based choice
            mcq_options.append(f"{choice}. {category}")
            choice_to_category[choice] = category
        
        # Join the MCQ options into a single string
        mcq_options_str = "\n".join(mcq_options)
        
        return mcq_options_str, choice_to_category
    
    def _append_answer_through_options(self, response, choice_to_category, index):
        answer_json = _parse_json(response)
        # Check length TODO: Log misalignment?
        if len(answer_json) > len(self.result_tuples):
            # Truncate, but should we put warning?
            answer_json = answer_json[:len(self.result_tuples)]
        elif len(answer_json) < len(self.result_tuples):
            # Unexpected, leaving rest to None, should we put warning?
            answer_json.extend([{self.answer_choice_fld: None}] * (len(self.result_tuples) - len(answer_json)))  
        
        # Modify answer
        for i, opt_answer in enumerate(answer_json):
            ans = opt_answer[self.answer_choice_fld]
            if ans is not None:
                ans = ans.upper()
                actual_answer = choice_to_category.get(ans, None)
            else:
                actual_answer = None
            if self.target_index[index] < len(self.result_tuples[i]):
                self.result_tuples[i][self.target_index[index]] = actual_answer
            elif self.target_index[index] == len(self.result_tuples[i]):
                self.result_tuples[i].append(actual_answer)
            else:
                raise IndexError("Unexpected indexing error when appending answer")
    
    def _convert_tuples_to_json(self, transform_index):
        column_names = [self.columns[j].column_name for j in self.projection_indices[transform_index]]
        json_array = []
        for tup in self.result_tuples:
            projected_tuple = [tup[j] for j in self.projection_indices[transform_index]]
            json_obj = {}
            for col_name, entry in zip(column_names, projected_tuple):
                json_obj[col_name] = entry
            json_array.append(json_obj)
            
        return json_array
    
    def _naive_mapping(self):
        """Clustering, but just prompt the whole thing"""

        prompt_system = """You are a helpful assistant that follows the instruction provided by the user."""
        
        for index, col in enumerate(self.transform_columns):
            # TODO: Edge case when K is greater or equal than the number of inputs (clustering is not possible); should we log warning?
            if col.k >= len(self.result_tuples):
                for i in range(len(self.result_tuples)):
                    if self.target_index[index] < len(self.result_tuples[i]):
                        self.result_tuples[i][self.target_index[index]] = None
                    elif self.target_index[index] == len(self.result_tuples[i]):
                        self.result_tuples[i].append(None)
                    else:
                        raise IndexError("Unexpected indexing error when appending answer")
            else:
                json_array = self._convert_tuples_to_json(index)
                
                # Clustering prompt
                messages_cluster = [
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": f"""
                    Without any explanation, given the following JSON array of objects, come up with {col.k} distinct categories.
                    Each category should represent a meaningful grouping based on this context: "{col.prompt_text}".
                    Output only the {col.k} category names separated by commas.
                    
                    JSON array:
                    {str(json_array)}
                    """}
                ]
                clustered_categories = self.client_model.complete_messages(messages_cluster, temperature=0.1)
                mcq_options_str, choice_to_category = self._generate_mcq_options(clustered_categories)

                # Classification through MCQ
                messages_classify = [
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": f"""
                    Without any explanation, given the following JSON array of objects, classify each object
                    by choosing only one from the given alphabet choices (e.g., A, B, C), based on the
                    following context: {col.prompt_text}

                    Output only the answers in JSON format with field '{self.answer_choice_fld}'.
                    For example:
                    [
                        {{"{self.answer_choice_fld}": "A"}},
                        {{"{self.answer_choice_fld}": "B"}}
                    ]

                    Options:
                    {mcq_options_str}

                    JSON array:
                    {str(json_array)}
                    """}
                ]
                response = self.client_model.complete_messages(messages_classify, temperature=0.1)
                self._append_answer_through_options(response, choice_to_category, index)
    
    def _get_projection_columns_indices(self):
        self.columns = self.children[0].columns
        column_name_to_index = {col.column_name: i for i, col in enumerate(self.columns)}

        for col in self.transform_columns:
            # Map original_columns to indices in self.columns
            proj_index = []
            for source_col in col.original_columns:
                if source_col in column_name_to_index:
                    proj_index.append(column_name_to_index[source_col])
                else:
                    raise RuntimeError(f"{source_col} not found in children columns")
            self.projection_indices.append(proj_index)

            # Determine target index for the transformed column
            if col.column_name in column_name_to_index:
                # Existing column
                self.target_index.append(column_name_to_index[col.column_name])
            else:
                # New column
                new_index = len(self.columns)
                self.target_index.append(new_index)
                self.columns.append(VirtualColumn(col.column_name))
                column_name_to_index[col.column_name] = new_index

    def next(self):
        """
        Right now we are not supporting K = None (adaptive clustering)
        Process each input tuple with semantic prompts
        Returns:
            Generator yielding processed tuples
        """
        if not self.has_init_models:
            self.has_init_models = True
            self.client_model = default_client_model()
            self.embedding_model = default_embedding_model()
        
        # If K is None or K is > 1, then get them all together before
        if len(self.result_tuples) == 0:
            embed_list = []
            
            for input_tuple in self.children[0].next():
                input_tuple = list(input_tuple)
                # Generate embedding based on required groupby specification
                # TODO: Allow generation embeddings per batch
                curr_row_embed = []
                for i in range(len(self.transform_columns)):
                    self.result_tuples.append(input_tuple)
                    projected_tuple = [input_tuple[j] for j in self.projection_indices[i]]
                    str_proj_tuple = ""
                    for j, entry in enumerate(projected_tuple):
                        col_index = self.projection_indices[i][j]
                        str_proj_tuple += f'{self.columns[col_index].column_name}: {entry}\n'
                    curr_row_embed.append(self.embedding_model.generate_embeddings(str_proj_tuple.strip()))
                    embed_list.append(curr_row_embed)

            if not self.need_clustering:
                self._transform_tuples()
            else:
                self._naive_mapping()

        for tup in self.result_tuples:
            yield tup

    def close(self):
        """Clean up resources"""
        if self.stream:
            try:
                self.stream.close()
            except:
                pass
        self.stream = None
        self.children[0].close()

class SemanticJoin(PhysicalOperator):
    def __init__(self, condition, join_type, children_table_names):
        """
        Args:
            schema: Schema of the table.
        """
        super().__init__('SemanticJoin')
        self.columns = []
        self.children_columns = []
        self.condition = condition.condition
        self.join_table_columns = condition.table_columns
        self.join_type = join_type
        self.children_table_names = children_table_names
        self.has_init_models = False
        self.client_model = None
        self.embedding_model = None
        self.tab_col_inds = [] # Format: (tuple of (child_index, column_index))

    def _get_proj_index(self):
        # Iterate over which children and which column index to be joined with; for now assume only left and right
        for join_col in self.join_table_columns:
            match_found = False
            for child_index, child_table_name in enumerate(self.children_table_names):
                if join_col.table_name == child_table_name:
                    for col_index, col in enumerate(self.children_columns[child_index]):
                        if join_col == col: # Compare TableColumn with TableColumn
                            self.tab_col_inds.append((child_index, col_index))
                            match_found = True
                            break

                    if match_found:
                        break
            
            # If no match was found for some reason
            if not match_found:
                raise ValueError(f"Column '{join_col.column_name}' not found in table '{join_col.table_name}'.")

    def open(self):
        if len(self.children) <= 1:
            raise ValueError("SemanticJoin requires more than one input operator")

        for child in self.children:
            child.open()
            self.columns.extend(child.columns)
            self.children_columns.append(child.columns)
        self._get_proj_index()
        super().open()
        
    def _construct_prompt_condition(self, left_row, right_row):
        # Condition is already a string that just needs to be formatted
        entry_vals = []
        for child_ind, col_ind in self.tab_col_inds:
            if child_ind == 0:
                entry_vals.append(left_row[col_ind])
            else:
                entry_vals.append(right_row[col_ind])
                
        return self.condition.format(*tuple(entry_vals))

    def next(self):
        """
        Process whole document with semantic prompts
        Returns:
            Dataframe
        """
        temperature = 0.1
        if not self.has_init_models:
            self.has_init_models = True
            self.client_model = default_client_model()
            self.embedding_model = default_embedding_model()
        
        cached_right_rows = []

        # TODO: Smarter batch inference call
        for left_row in self.children[0].next():
            if len(cached_right_rows) == 0:
                # Materialize right row
                for right_row in self.children[1].next():
                    cached_right_rows.append(right_row)
            
            for right_row in cached_right_rows:
                msg = self._construct_prompt_condition(left_row, right_row)
                messages = [
                    {"role": "system", "content": "You are a strict judge that outputs only 'true' or 'false' with no punctuation or extra characters based on a given statement."},
                    {"role": "user", "content": msg}
                ]            
                response = self.client_model.complete_messages(messages, temperature=temperature)
                if response.lower() == "true":
                    yield left_row + right_row

    def close(self):
        """Clean up resources"""
        for child in self.children:
            child.close()
        super().close()

class SemanticScan(PhysicalOperator):
    """Physical operator for processing document into a proper table with prompts"""

    def __init__(self, target_columns, prompt_columns, filter=None):
        """
        Args:
            schema: Schema of the table.
        """
        super().__init__('SemanticScan')
        self.columns = target_columns
        self._convert_prompt_columns_to_schema(prompt_columns)
        self.has_init_models = False
        self.client_model = None
        self._filter = filter
        self.document = None
        self.embeddings = None
        self.stream = None
        self.result_tuples = None

    def open(self):
        if len(self.children) != 1:
            raise ValueError("SemanticScan requires exactly one input operator")
        self.children[0].open()
        
        if self._filter:
            if isinstance(self._filter, Filter):
                columns = []
                type_oids = []
                for table_column in self.columns:
                    columns.append(table_column)
                self._filter.set_tuple_columns(columns, type_oids=None)
            elif isinstance(self._filter, SemanticFilter):
                self._filter.set_proj_index(self.columns)
        super().open()
        
    def _convert_prompt_columns_to_schema(self, prompt_columns):
        # Extract schema from PromptColumns
        schema_lines = []
        
        for column in prompt_columns:
            if isinstance(column, PromptColumn):
                schema_lines.append(f"{column.column_name}: Extract '{column.prompt_text.value}'")

        # Combine into a formatted schema
        self.schema = "\n".join(schema_lines)
        
    def _parse_json_into_tuples(self, raw_output):
        cleaned_json = _parse_json(raw_output)
        if len(cleaned_json) == 0:
            return []

        # Extract column names (keys of the first dictionary)
        table_tuples = [tuple(item.get(col.column_name, None) for col in self.columns) for item in cleaned_json]

        return table_tuples

    def next(self):
        """
        Process whole document with semantic prompts
        Returns:
            Dataframe
        """
        if not self.has_init_models:
            self.has_init_models = True
            self.client_model = default_client_model()
        
        if self.result_tuples is None:
            temperature = 0.1
            self.document = ""
            for doc, embed in self.children[0].next():
                self.document += doc  # Assume child is Scan; only take the document, not embedding
                self.document += "\n"
                

            prompt_system = """
            You are a data extraction assistant.
            Your task is to extract structured information from unstructured text and format it into JSON.
            Follow the provided schema exactly.
            """

            messages = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": f"""
                Convert the following raw text into a JSON array using this schema:
                {self.schema}
                
                Ensure the response starts with '[' and ends with ']', with no additional text or explanation.
                Missing or empty values should be represented as null. 
                
                Raw text:
                {self.document}
                """}
            ]
            response = self.client_model.complete_messages(messages, temperature=temperature)
            self.result_tuples = self._parse_json_into_tuples(response)
                
        if self._filter:
            for filtered_tup in self._filter.filter(iter(self.result_tuples)):
                yield filtered_tup
        else:
            for tup in self.result_tuples:
                yield tup

    def close(self):
        """Clean up resources"""
        self.children[0].close()
        super().close()
