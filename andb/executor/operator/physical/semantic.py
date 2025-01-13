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

def _parse_json_into_tuples(raw_output):
    cleaned_json = _parse_json(raw_output)
    if len(cleaned_json) == 0:
        return []

    # Extract column names (keys of the first dictionary)
    columns = list(cleaned_json[0].keys())
    table_tuples = [tuple(item.get(col, None) for col in columns) for item in cleaned_json]

    return table_tuples

def _parse_markdown_into_tuples(raw_markdown):
    """
    I think this should be deprecated since it reduces performance, although it saves number of output tokens
    Parses a markdown-style table into a list of tuples.
    The assumption is that `|` is used as the separator.
    """
    # Clean and split lines and combine into a CSV string
    lines = raw_markdown.strip().split("\n")
    cleaned_lines = [line.strip("|").strip() for line in lines]

    # Convert each line into a list of cells
    table_tuples = []
    for line in cleaned_lines:
        # Split on '|' and strip spaces
        cells = [cell.strip() for cell in line.split("|")]

        # Replace any cell with only dashes with None (to simulate NaN)
        cells = [None if re.fullmatch(r"-+", cell) else cell for cell in cells]

        # Append as a tuple
        table_tuples.append(tuple(cells))

    # Remove the separator row (if exists)
    if len(table_tuples) > 1 and all(cell is None for cell in table_tuples[0]):
        table_tuples.pop(0)
    if len(table_tuples) > 1 and all(cell is None for cell in table_tuples[-1]):
        table_tuples.pop()

    return table_tuples

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
        if isinstance(condition, Condition):
            self.condition_prompt = f'we only consider the following condition: {str(condition)}'
        elif isinstance(condition, str):
            self.condition_prompt = f'we only consider the following condition: {condition}'
        else:
            raise ValueError("Condition must be a Condition object or a string")

    def open(self):
        return super().open()

    def close(self):
        return super().close()

    def judge(self, tuple):
        # Convert tuple to text format for analysis
        text = str(tuple[0]) if len(tuple) == 1 else " ".join(str(x) for x in tuple)

        try:
            # Create a prompt that combines the condition and the tuple text
            prompt = f"""
            Condition: {self.condition_prompt}
            Text to evaluate: {text}
            
            Does the text satisfy the condition? Please respond with only 'true' or 'false'.
            """

            # Call OpenAI API
            messages = [
                {"role": "system", "content": "You are a precise evaluator that only responds with 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ]

            response = self.client_model.complete_messages(messages=messages, temperature=0.1)

            # Get the response and convert to boolean
            result = response.strip().lower()
            return result == 'true'

        except Exception as e:
            logging.error(f"Error in semantic filtering: {e}")
            raise e

    def next(self):
        if not self.has_init_models:
            self.has_init_models = True
            self.client_model = default_client_model()
        
        for tuple in self.children[0].next():
            if self.judge(tuple):
                yield tuple


class SemanticJoin(PhysicalOperator):
    """
    Semantic Join operator that uses OpenAI API to join documents based on their semantic meaning
    """

    def __init__(self, join_type, target_columns=None, join_filter: Filter = None):
        super().__init__('SemanticJoin')
        self.has_init_models = False
        self.client_model = None
        self.join_type = join_type
        self.target_columns = target_columns
        self.join_filter = join_filter
        self.join_prompt = ''

        if self.join_type == JoinType.INNER_JOIN:
            self.join_prompt += "Only return the relationship if the two texts are semantically related."
        elif self.join_type == JoinType.LEFT_JOIN:
            self.join_prompt += "Only return the relationship if the 'Text 1' is semantically related to the Text 2."
        elif self.join_type == JoinType.RIGHT_JOIN:
            self.join_prompt += "Only return the relationship if the 'Text 2' is semantically related to the Text 1."
        elif self.join_type == JoinType.FULL_JOIN:
            self.join_prompt += "Return the relationship if the two texts are semantically related."

    def open(self):
        """
        Initialize the operator and validate children
        """
        # Validate we have exactly 2 children (left and right input)
        if len(self.children) != 2:
            raise ValueError("SemanticJoin requires exactly two input operators")

        # Open both child operators
        self.children[0].open()
        self.children[1].open()

        # Set output columns
        self.columns = (
                self.children[0].columns +  # Left input columns
                self.children[1].columns +  # Right input columns
                [TableColumn(DummyTableName.TEMP_TABLE_NAME, 'relationship')]  # Add relationship column
        )

    def next(self):
        """
        Generate joined results by semantically comparing documents
        """
        if not self.has_init_models:
            self.has_init_models = True
            self.client_model = default_client_model()
        
        # Get all texts from left child
        for left_tuple in self.children[0].next():
            left_text = left_tuple[0]  # Assuming text content is first column

            # Get all texts from right child
            for right_tuple in self.children[1].next():
                right_text = right_tuple[0]  # Assuming text content is first column

                # Get semantic relationship using OpenAI
                relationship = self._get_semantic_relationship(left_text, right_text)

                # Yield combined tuple with relationship
                yield left_tuple + right_tuple + (relationship,)

    def _get_semantic_relationship(self, text1, text2):
        """
        Use OpenAI API to analyze relationship between two texts
        """
        try:
            # Create prompt combining both texts
            prompt = f"""
            Text 1: {text1}
            
            Text 2: {text2}
            
            {self.join_prompt}
            """

            messages = [
                {"role": "system",
                 "content": "You are a text analysis expert focused on finding relationships between documents."},
                {"role": "user", "content": prompt}
            ]

            # Call OpenAI API
            # Lower temperature for more focused responses and limit response length
            response = self.client_model.complete_messages(messages=messages, temperature=0.3, max_tokens=200)

            # Extract and return the relationship description
            return response.strip()

        except Exception as e:
            raise ExecutionStageError(f"Error in semantic analysis: {e}")

    def close(self):
        """
        Clean up resources
        """
        self.children[0].close()
        self.children[1].close()
        super().close()


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


class SemanticScan(PhysicalOperator):
    """Physical operator for processing document into a proper table with prompts"""

    def __init__(self, target_columns, prompt_columns, intermediate_data="tabular"):
        """
        Args:
            schema: Schema of the table.
            intermediate_data: Intermediate output (for debugging purposes between 'json' and 'tabular')
        """
        super().__init__('SemanticScan')
        self.columns = target_columns
        self._convert_prompt_columns_to_schema(prompt_columns)
        self.has_init_models = False
        self.client_model = None
        self.intermediate_data = intermediate_data
        if self.intermediate_data not in ["json", "tabular"]:
            raise NotImplementedError(f"Intermediate data `{self.intermediate_data}` is not implemented!")

        self.document = None
        self.embeddings = None
        self.stream = None
        self.result_tuples = None

    def open(self):
        if len(self.children) != 1:
            raise ValueError("SemanticScan requires exactly one input operator")
        self.children[0].open()
        super().open()
        
    def _convert_prompt_columns_to_schema(self, prompt_columns):
        # Extract schema from PromptColumns
        schema_lines = []
        
        for column in prompt_columns:
            if isinstance(column, PromptColumn):
                schema_lines.append(f"{column.column_name}: Extract '{column.prompt_text}'")

        # Combine into a formatted schema
        self.schema = "\n".join(schema_lines)

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
                
            if self.intermediate_data == 'json':
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
                self.result_tuples = _parse_json_into_tuples(response)

            else:
                prompt_system = """
                You are a data extraction assistant.
                Your task is to extract structured information from unstructured text and format it into a row-based tabular format.
                Follow the provided schema exactly and ensure the output adheres to the specified structure.
                """

                messages = [
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": f"""
                    Convert the following raw text into a row-based tabular format with `|` as the delimitter, and use this schema:
                    {self.schema}
                    
                    Do not include any additional text or explanation outside the table.
                    
                    Raw text:
                    {self.document}
                    """}
                ]
                response = self.client_model.complete_messages(messages, temperature=temperature)
                self.result_tuples = _parse_markdown_into_tuples(response)

        for tup in self.result_tuples:
            yield tup

    def close(self):
        """Clean up resources"""
        self.children[0].close()
        super().close()
