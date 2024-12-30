from threading import local

from andb.catalog.oid import OID_DATABASE_ANDB
from andb.constants.macros import INVALID_XID

# For user configuration during the session
class SessionParameter(local):
    llm = 'openai'
    openai_model = 'gpt-4o-mini'
    openai_api_key = None

# TODO: refactor this class and fix the runtime error
class SessionVars(local):
    database_oid = OID_DATABASE_ANDB
    session_xid = None

