import struct
from collections import OrderedDict

# big endian is friendly to keep order
CTYPE_BIG_ENDIAN = '>'

CTYPE_TYPE_CHAR = 'c'
CTYPE_TYPE_CHAR_ARRAY = 's'
CTYPE_TYPE_PASCAL_STRING = 'p'
CTYPE_TYPE_SHORT = 'H'
CTYPE_TYPE_INT4 = 'i'
CTYPE_TYPE_UINT4 = 'I'
CTYPE_TYPE_INT8 = 'q'
CTYPE_TYPE_UINT8 = 'Q'
CTYPE_TYPE_FLOAT4 = 'f'
CTYPE_TYPE_FLOAT8 = 'd'
CTYPE_TYPE_BOOL = '?'


class Field:
    def __init__(self, name, ctype, num, default):
        self.name = name
        self.ctype = ctype
        self.pretty_ctype = None
        self.num = num
        self.default = default

        assert ctype
        assert self.num >= 0

    def __repr__(self):
        if self.num == 1:
            return '<%s %s>' % (
                self.pretty_ctype or self.ctype, self.name)
        else:
            return '<%s %s[%d]>' % (
                self.pretty_ctype or self.ctype, self.name, self.num
            )


class CharField(Field):
    def __init__(self, name=None, num=1):
        """It is always unsigned."""
        if num > 1:
            # default value is an empty char array
            super(CharField, self).__init__(name, CTYPE_TYPE_CHAR_ARRAY, num, bytes(num))
        else:
            # short type is easier than char for only one element
            super(CharField, self).__init__(name, CTYPE_TYPE_SHORT, num, 0)
        self.pretty_ctype = 'char'


class TextField(Field):
    def __init__(self, name=None, num=1):
        # default value is an empty char array
        super(TextField, self).__init__(name, CTYPE_TYPE_CHAR_ARRAY, 1, bytes(0))
        self.pretty_ctype = 'text'


class Integer4Field(Field):
    def __init__(self, name=None, unsigned=False, num=1):
        if unsigned:
            super(Integer4Field, self).__init__(name, CTYPE_TYPE_UINT4, num, 0)
            self.pretty_ctype = 'unsigned int'
        else:
            super(Integer4Field, self).__init__(name, CTYPE_TYPE_INT4, num, 0)
            self.pretty_ctype = 'int'


class Integer8Field(Field):
    def __init__(self, name=None, unsigned=False, num=1):
        if unsigned:
            super(Integer8Field, self).__init__(name, CTYPE_TYPE_UINT8, num, 0)
            self.pretty_ctype = 'unsigned long long'
        else:
            super(Integer8Field, self).__init__(name, CTYPE_TYPE_INT8, num, 0)
            self.pretty_ctype = 'long long'


class Float4Field(Field):
    def __init__(self, name=None, num=1):
        super(Float4Field, self).__init__(name, CTYPE_TYPE_FLOAT4, num, 0.)
        self.pretty_ctype = 'float'


class Float8Field(Field):
    def __init__(self, name=None, num=1):
        super(Float8Field, self).__init__(name, CTYPE_TYPE_FLOAT8, num, 0.)
        self.pretty_ctype = 'double'


class StructureMeta(type):
    def __new__(cls, name, bases, attrs):
        if name == 'CStructure':
            return type.__new__(cls, name, bases, attrs)

        # should be in order
        mappings = OrderedDict()
        format_ = list()
        for k, v in attrs.items():
            if isinstance(v, CStructure):
                raise NotImplementedError('Not supported to set non-primitive data types yet.')
            if isinstance(v, Field):
                # set key name as the default name
                if not v.name:
                    v.name = k
                mappings[k] = v
                # this is an array
                if v.num > 1:
                    format_.append(str(v.num))
                format_.append(v.ctype)

        attrs['__mappings__'] = mappings
        attrs['__cformat__'] = CTYPE_BIG_ENDIAN + ''.join(format_)
        return type.__new__(cls, name, bases, attrs)


class CStructure(metaclass=StructureMeta):
    __mappings__ = None
    __cformat__ = None

    def pack(self) -> bytes:
        values = list()
        for k, field in self.__mappings__.items():
            real_value = getattr(self, k)
            # append default values if not set
            if isinstance(real_value, Field):
                values.append(field.default)
            else:
                if field.num > 1:
                    if isinstance(real_value, list):
                        values.extend(real_value)
                    elif isinstance(real_value, bytes):
                        values.append(real_value)
                    else:
                        raise ValueError('%s should be a list or bytes.' % k)

                else:
                    # We don't receive an array contains one element!
                    values.append(real_value)
        try:
            return struct.pack(self.__cformat__, *values)
        except struct.error as e:
            raise struct.error("%s. format is '%s' and values are %s." % (
                e, self.__cformat__, values))

    def unpack(self, buffer: bytes):
        values = struct.unpack(self.__cformat__, buffer)
        if len(values) == 0:
            return

        i = 0
        for k, field in self.__mappings__.items():
            if field.num > 1 and not isinstance(field, CharField):
                array_ = values[i: i + field.num]
                i += field.num
                setattr(self, k, array_)
            else:
                value = values[i]
                setattr(self, k, value)
                i += 1

    @classmethod
    def size(cls) -> int:
        return struct.calcsize(cls.__cformat__)

    def __eq__(self, other) -> bool:
        if not isinstance(other, CStructure):
            return False
        return (self.__cformat__ == self.__cformat__ and
                self.pack() == other.pack())

    def __hash__(self) -> int:
        return hash((self.__cformat__, self.pack()))

    def __len__(self) -> int:
        return self.size()

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_')])})"


def pack(fmt_flag, *v):
    return struct.pack(f'{CTYPE_BIG_ENDIAN}{fmt_flag}', *v)

def unpack(fmt_flag, b):
    return struct.unpack(f'{CTYPE_BIG_ENDIAN}{fmt_flag}', b)

def unpack_one(fmt_flag, b):
    return struct.unpack(f'{CTYPE_BIG_ENDIAN}{fmt_flag}', b)[0]

def calcsize(fmt_flag):
    return struct.calcsize(f'{CTYPE_BIG_ENDIAN}{fmt_flag}')
