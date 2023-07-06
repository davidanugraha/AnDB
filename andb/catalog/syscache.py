from .attribute import _ANDB_ATTRIBUTE
from .class_ import _ANDB_CLASS
from .database import _ANDB_DATABASE
from .type import _ANDB_TYPE

CATALOG_ANDB_ATTRIBUTE = _ANDB_ATTRIBUTE
CATALOG_ANDB_CLASS = _ANDB_CLASS
CATALOG_ANDB_DATABASE = _ANDB_DATABASE
CATALOG_ANDB_TYPE = _ANDB_TYPE

_ALL_CATALOGS = (CATALOG_ANDB_ATTRIBUTE, CATALOG_ANDB_CLASS,
                 CATALOG_ANDB_DATABASE, CATALOG_ANDB_TYPE)


def get_all_catalogs():
    return _ALL_CATALOGS
