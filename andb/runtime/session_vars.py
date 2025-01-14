from threading import local

from andb.catalog.oid import OID_DATABASE_ANDB
from andb.constants.macros import INVALID_XID

# For user configuration during the session
class SessionParameter(local):
    # Client model related parameters
    client_llm = 'openai'
    client_openai_model = 'gpt-4o-mini'
    client_hf_repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    client_offline_model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
    
    # Embedding model related parameters
    embed_llm = 'openai'
    embed_openai_model = 'text-embedding-3-large'
    embed_hf_repo_id = 'sentence-transformers/all-MiniLM-L6-v2'
    embed_offline_model_path = 'sentence-transformers/all-MiniLM-L6-v2'
    
    hf_token = None
    openai_api_key = None

# TODO: refactor this class and fix the runtime error
class SessionVars(local):
    database_oid = OID_DATABASE_ANDB
    session_xid = None

