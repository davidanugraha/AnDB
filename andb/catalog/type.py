from functools import partial

from andb.common import hash_functions
from andb.common import cstructure
from ._base import CatalogForm, CatalogTable
from .oid import INVALID_OID, OID_SYSTEM_TABLE_TYPE, OID_TYPE_START, OID_TYPE_END
from andb.common.utils import memoize
from andb.constants.strings import BIG_END
from math import sqrt

VARIABLE_LENGTH = 0
NULL_LENGTH = -1
VARIABLE_TYPE_HEADER_LENGTH = 4  # int4
_VARIABLE_TYPE_CTYPE = cstructure.CTYPE_TYPE_INT4


def generic_cmp(a, b):
    return a - b


class AndbBaseType:
    oid = INVALID_OID
    type_name = 'undefined'
    type_alias = ''
    type_bytes = VARIABLE_LENGTH
    type_char = 'x'
    type_default = 0
    hash_func = None

    @classmethod
    def to_bytes(cls, v):
        return cstructure.pack(cls.type_char, v)

    @classmethod
    def to_datum(cls, b):
        return cstructure.unpack_one(cls.type_char, b)

    @staticmethod
    def cast_to_string(v):
        return str(v)

    @staticmethod
    def cast_from_string(v):
        raise NotImplementedError()

    @classmethod
    def bytes_length(cls, b):
        if b is None:
            return NULL_LENGTH
        return cls.type_bytes

    @classmethod
    def format_value(cls, value):
        return value


class IntegerType(AndbBaseType):
    oid = 1000
    type_name = 'integer'
    type_alias = 'int'
    type_bytes = 4
    type_char = cstructure.CTYPE_TYPE_INT4
    type_default = 0
    hash_func = partial(hash_functions.hash_int, length=4)

    @staticmethod
    def cast_from_string(v):
        return int(v)

    @classmethod
    def to_bytes(cls, v):
        # order-preserving for byte encoding
        v += 0xffffffff >> 1
        return int.to_bytes(v, length=cls.type_bytes, byteorder=BIG_END, signed=False)

    @classmethod
    def to_datum(cls, b):
        v = int.from_bytes(b, byteorder=BIG_END, signed=False)
        v -= 0xffffffff >> 1
        return v


class BigintType(AndbBaseType):
    oid = 1001
    type_name = 'bigint'
    type_bytes = 8
    type_char = cstructure.CTYPE_TYPE_INT8
    type_default = 0
    hash_func = partial(hash_functions.hash_int, length=8)

    @staticmethod
    def cast_from_string(v):
        return int(v)

    @classmethod
    def to_bytes(cls, v):
        # order-preserving for byte encoding
        v += 0xffffffffffffffff >> 1
        return int.to_bytes(v, length=cls.type_bytes, byteorder=BIG_END, signed=False)

    @classmethod
    def to_datum(cls, b):
        v = int.from_bytes(b, byteorder=BIG_END, signed=False)
        v -= 0xffffffffffffffff >> 1
        return v


class RealType(AndbBaseType):
    oid = 1002
    type_name = 'real'
    type_alias = 'float'
    type_bytes = 4
    type_char = cstructure.CTYPE_TYPE_FLOAT4
    type_default = 0.
    hash_func = partial(hash_functions.hash_float, length=4)

    @staticmethod
    def cast_from_string(v):
        return float(v)


class DoubleType(AndbBaseType):
    oid = 1003
    type_name = 'double precision'
    type_alias = 'double'
    type_bytes = 8
    type_char = cstructure.CTYPE_TYPE_FLOAT8
    type_default = 0.
    hash_func = partial(hash_functions.hash_float, length=8)

    @staticmethod
    def cast_from_string(v):
        return int(v)


class BooleanType(AndbBaseType):
    oid = 1003
    type_name = 'boolean'
    type_alias = 'bool'
    type_bytes = 1
    type_char = cstructure.CTYPE_TYPE_BOOL
    type_default = False
    hash_func = hash_functions.hash_bool

    @staticmethod
    def cast_to_string(v):
        return 'true' if v else 'false'

    @staticmethod
    def cast_from_string(v):
        return v.lower() == 'true'


class CharType(AndbBaseType):
    oid = 1005
    type_name = 'char'
    type_bytes = 1
    type_char = cstructure.CTYPE_TYPE_CHAR
    type_default = '\0'
    hash_func = hash_functions.hash_string

    @classmethod
    def to_bytes(cls, v):
        if isinstance(v, int):
            v = chr(v)
        encoded_v = str.encode(v, encoding='utf8')
        return cstructure.pack(f'{len(encoded_v)}{cls.type_char}', encoded_v)

    @classmethod
    def to_datum(cls, b):
        return cstructure.unpack_one(f'{len(b)}{cls.type_char}', b).decode(encoding='utf8')

    @staticmethod
    def cast_from_string(v):
        return v


class VarcharType(AndbBaseType):
    oid = 1006
    type_name = 'varchar'
    type_bytes = VARIABLE_LENGTH
    type_char = cstructure.CTYPE_TYPE_CHAR_ARRAY
    type_default = ''
    hash_func = hash_functions.hash_string

    # notice: must be truncated ahead
    @classmethod
    def to_bytes(cls, v):
        encoded_v = str.encode(v, encoding='utf8')
        return cstructure.pack(f'{len(encoded_v)}{cls.type_char}', encoded_v)

    @classmethod
    def to_datum(cls, b):
        return cstructure.unpack_one(f'{len(b)}{cls.type_char}', b).decode(encoding='utf8')

    @staticmethod
    def cast_from_string(v):
        return v

    @classmethod
    def bytes_length(cls, b):
        if b is None:
            return NULL_LENGTH
        return len(b)


class TextType(AndbBaseType):
    oid = 1007
    type_name = 'text'
    type_bytes = VARIABLE_LENGTH
    type_char = cstructure.CTYPE_TYPE_CHAR_ARRAY
    type_default = ''
    hash_func = hash_functions.hash_string

    @classmethod
    def to_bytes(cls, v):
        encoded_v = str.encode(v, encoding='utf8')
        return (cstructure.pack(_VARIABLE_TYPE_CTYPE, len(encoded_v)) +
                cstructure.pack(f'{len(encoded_v)}{cls.type_char}', encoded_v))

    @classmethod
    def to_datum(cls, b):
        assert len(b) >= VARIABLE_TYPE_HEADER_LENGTH
        b_length = cstructure.unpack_one(_VARIABLE_TYPE_CTYPE, b[:VARIABLE_TYPE_HEADER_LENGTH])
        b_content = b[VARIABLE_TYPE_HEADER_LENGTH: VARIABLE_TYPE_HEADER_LENGTH + b_length]
        return cstructure.unpack_one(f'{len(b_content)}{cls.type_char}', b_content).decode(encoding='utf8')

    @staticmethod
    def cast_from_string(v):
        return v

    @classmethod
    def bytes_length(cls, b):
        if b is None:
            return NULL_LENGTH
        assert len(b) >= VARIABLE_TYPE_HEADER_LENGTH
        b_length = cstructure.unpack_one(_VARIABLE_TYPE_CTYPE, b[:VARIABLE_TYPE_HEADER_LENGTH])
        return b_length


class VectorType(AndbBaseType):
    oid = 1008
    type_name = 'vector'
    type_bytes = VARIABLE_LENGTH
    type_char = cstructure.CTYPE_TYPE_FLOAT8
    type_default = []
    hash_func = hash_functions.hash_array

    @classmethod
    def to_bytes(cls, v):
        if isinstance(v, str):
            v = cls.cast_from_string(v)
        assert isinstance(v, list) and all(isinstance(i, float) for i in v), \
            "v must be a list of floats"
        # v is a list of floats
        length = len(v)
        # Pack the length as an int4
        packed_length = cstructure.pack(_VARIABLE_TYPE_CTYPE, length)
        # Pack the float values
        format_string = f'{length}{cls.type_char}'
        packed_floats = cstructure.pack(format_string, *v)
        return packed_length + packed_floats

    @classmethod
    def to_datum(cls, b):
        # Unpack the length
        length = cstructure.unpack_one(_VARIABLE_TYPE_CTYPE, b[:VARIABLE_TYPE_HEADER_LENGTH])
        # Unpack the floats
        format_string = f'{length}{cls.type_char}'
        floats = cstructure.unpack(format_string, b[VARIABLE_TYPE_HEADER_LENGTH:])
        return list(floats)

    @staticmethod
    def cast_from_string(v):
        # Assume the string representation is like '[1.0, 2.0, 3.0]'
        v = v.strip('[]').split(',')
        return [float(x.strip()) for x in v]

    @classmethod
    def bytes_length(cls, b):
        if b is None:
            return NULL_LENGTH
        length = cstructure.unpack_one(_VARIABLE_TYPE_CTYPE, b[:VARIABLE_TYPE_HEADER_LENGTH])
        return VARIABLE_TYPE_HEADER_LENGTH + length * cstructure.calcsize(cls.type_char)

    @classmethod
    def format_value(cls, value):
        vector = cls.cast_from_string(value)
        return f'[{", ".join(map(str, vector))}]'


class AndbTypeForm(CatalogForm):
    __fields__ = {
        'oid': 'bigint',
        'type_name': 'text',
        'type_alias': 'text',
        'type_bytes': 'integer',
        'type_char': 'char'
    }

    def __init__(self, defined_type):
        self.oid = defined_type.oid
        self.type_name = defined_type.type_name
        self.type_alias = defined_type.type_alias
        self.type_bytes = defined_type.type_bytes
        self.type_char = defined_type.type_char
        self.type_default = defined_type.type_default

    def __lt__(self, other):
        return self.oid < other.oid


_BUILTIN_TYPES = (
    IntegerType, BigintType, RealType, DoubleType,
    BooleanType, CharType, VarcharType, TextType,
    VectorType
)

_BUILTIN_TYPES_DICT = {i.type_name: i for i in _BUILTIN_TYPES}


class AndbTypeTable(CatalogTable):
    __tablename__ = 'andb_type'
    __oid__ = OID_SYSTEM_TABLE_TYPE
    __form__ = AndbTypeForm

    def init(self):

        for t in _BUILTIN_TYPES:
            self.insert(AndbTypeForm(t))

    def __init__(self):
        super().__init__()
        self._lookup_cache = {}

    def get_type_form(self, name):
        if len(self._lookup_cache) == 0:
            for r in self.rows:
                self._lookup_cache[r.type_name] = r
                if r.type_alias != '':
                    self._lookup_cache[r.type_alias] = r
        r = self._lookup_cache[name]
        return _BUILTIN_TYPES_DICT[r.type_name]
    
    @memoize
    def get_type_form_by_oid(self, oid):
        for r in self.rows:
            if r.oid == oid:
                return _BUILTIN_TYPES_DICT[r.type_name]
        return None

    def get_type_oid(self, name):
        meta = self.get_type_form(name)
        return meta.oid if meta else INVALID_OID

    @memoize
    def get_type_name(self, oid):
        assert isinstance(oid, int) and OID_TYPE_START <= oid <= OID_TYPE_END, "invalid oid"
        for r in self.rows:
            if r.oid == oid:
                return r.type_name

    def cast_datum_to_bytes(self, type_name, datum):
        meta = self.get_type_form(type_name)
        return meta.to_bytes(datum)

    def cast_bytes_to_datum(self, type_name, bytes_):
        meta = self.get_type_form(type_name)
        return meta.to_datum(bytes_)

def cast_value(value, destination_type_name):
    meta: AndbBaseType = _ANDB_TYPE.get_type_form(destination_type_name)
    if isinstance(value, str):
        return meta.cast_from_string(value)
    elif isinstance(value, type(meta.type_default)):
        # value seems already in the destination type
        return value
    else:
        raise NotImplementedError(f"Unsupported value type: {type(value)}")

_ANDB_TYPE = AndbTypeTable()
