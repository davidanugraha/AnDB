import logging
import os

from andb.catalog.class_ import RelationKinds
from andb.catalog.oid import OID_DATABASE_ANDB, INVALID_OID
from andb.catalog.syscache import CATALOG_ANDB_CLASS, CATALOG_ANDB_TYPE, CATALOG_ANDB_ATTRIBUTE, CATALOG_ANDB_DATABASE, \
    CATALOG_ANDB_INDEX
from andb.catalog.type import VARIABLE_LENGTH, VARIABLE_TYPE_HEADER_LENGTH, VarcharType
from andb.common.file_operation import directio_file_open, file_touch, file_size, file_close, file_write, file_open, \
    file_remove, file_read, file_lseek
from andb.common.utils import filesize_to_pageno
from andb.constants.filename import BASE_DIR
from andb.constants.strings import BIG_END
from andb.constants.values import MAX_TABLE_COLUMNS, PAGE_SIZE
from andb.errno.errors import RollbackError, DDLException
from andb.runtime import global_vars
from andb.storage.engines.heap.page import INVALID_BYTES
from andb.storage.engines.heap.page import INVALID_ITEM_ID
from andb.storage.engines.heap.bptree import BPlusTree, TuplePointer, create_node
from andb.storage.engines.heap.redo import WALAction, WALRecord
from andb.storage.engines.heap.undo import UndoOperation, UndoRecord
from andb.storage.lock import rlock
from andb.storage.utils import easy_tuple_serialize


class BufferedBPTree(BPlusTree):
    def __init__(self, relation):
        file_lseek(relation.fd, offset=0)
        header_size = BPlusTree.Header.size()
        root_pageno = self.deserialize_header(file_read(relation.fd, header_size)).root_pageno
        assert root_pageno >= 0
        buffer_page = global_vars.buffer_manager.get_page(relation, root_pageno)
        global_vars.buffer_manager.pin_page(buffer_page)

        super().__init__(root_node=buffer_page.page)
        self.relation = relation
        self.dirty_pageno = []

    def load_page(self, pageno):
        node = global_vars.buffer_manager.get_page(self.relation, pageno).page
        assert node.get_pageno() == pageno
        return node

    def _allocate_node(self, is_leaf):
        node = super()._allocate_node(is_leaf)
        buffer_page = global_vars.buffer_manager.create_buffer_page(self.relation, node.get_pageno(), node)
        buffer_page.mark_dirty()  # new node must be dirty
        return node

    def _need_to_split(self, node):
        #TODO: user-defined load factor
        return super()._need_to_split(node)


__relcache = {}


class Relation:
    def __init__(self, oid, database_oid, name):
        self.oid = oid
        self.database_oid = database_oid
        self.name = name
        self.attrs = ()
        self.file_path = os.path.join(BASE_DIR, str(database_oid), str(oid))
        self.refcount = 0
        self.kind = None
        self._last_pageno = None

    @property
    def is_heap(self):
        return self.kind == RelationKinds.HEAP_TABLE

    @property
    def fd(self):
        assert self.file_path
        return directio_file_open(self.file_path, os.O_RDWR | os.O_CREAT)

    def last_pageno(self):
        assert self.is_heap
        # if disk-based page is already full, buffer pool will
        # allocated a new page in memory. In that case, we marked allocated page to
        # `self._last_pageno` so return it directly is enough.
        if self._last_pageno:
            return self._last_pageno
        # pageno starts from 0
        return filesize_to_pageno(file_size(self.fd))

    def increase_last_pageno(self):
        self._last_pageno = self.last_pageno() + 1

    def __repr__(self):
        if self.oid == INVALID_OID:
            return '<InvalidRelation>'
        return str(self.oid)

    def __hash__(self):
        return self.oid

    def __eq__(self, other):
        return (isinstance(other, Relation) and
                other.oid == self.oid)


class TupleData:
    NULLS_BYTES = 8

    def __init__(self, python_tuple):
        self.python_tuple = python_tuple

    def __repr__(self):
        return str(self.python_tuple)

    def to_bytes(self, andb_attr_form_array):
        assert len(andb_attr_form_array) == len(self.python_tuple)
        nulls = 0x0000000000000000

        byte_array = bytearray()
        for i, form in enumerate(andb_attr_form_array):
            datum = self.python_tuple[i]
            if datum is None:
                # update nulls
                if form.notnull:
                    raise RollbackError(f'the field {form.name} cannot be null.')

                assert i < MAX_TABLE_COLUMNS
                nulls |= (1 << i)
            else:
                # as varchar is fixed length, we should truncate it here
                if form.type_oid == VarcharType.oid:
                    datum = datum[:form.length]
                bytes_ = CATALOG_ANDB_TYPE.cast_datum_to_bytes(
                    CATALOG_ANDB_TYPE.get_type_name(form.type_oid),
                    datum)
                byte_array.extend(bytes_)

        # concatenate nulls and data
        nulls_bytes = int.to_bytes(nulls, length=self.NULLS_BYTES, byteorder=BIG_END)
        return nulls_bytes + bytes(byte_array)

    @classmethod
    def from_bytes(cls, data, andb_attr_form_array):
        if len(data) < cls.NULLS_BYTES:
            raise RollbackError('the data corrupt')
        nulls = int.from_bytes(data[:cls.NULLS_BYTES], byteorder=BIG_END)
        values = []
        bytes_cursor = cls.NULLS_BYTES  # starts from real data
        for i, attr_form in enumerate(andb_attr_form_array):
            if nulls & (1 << i) == (1 << i):
                values.append(None)
            else:
                type_name = CATALOG_ANDB_TYPE.get_type_name(attr_form.type_oid)
                type_form = CATALOG_ANDB_TYPE.get_type_form(type_name)
                length = attr_form.length
                if length == VARIABLE_LENGTH:
                    length = type_form.bytes_length(
                        data[bytes_cursor: bytes_cursor + VARIABLE_TYPE_HEADER_LENGTH]) + VARIABLE_TYPE_HEADER_LENGTH
                value = CATALOG_ANDB_TYPE.cast_bytes_to_datum(
                    type_name, data[bytes_cursor: bytes_cursor + length]
                )
                values.append(value)
                bytes_cursor += length
        return cls(tuple(values))


# def search_relation(relation_name, database_name, kind):
#     results = CATALOG_ANDB_DATABASE.search(lambda r: r.name == database_name)
#     if len(results) != 1:
#         raise RollbackError('not found the database.')
#
#     database_oid = results[0].oid
#     return CATALOG_ANDB_CLASS.get_relation_oid(relation_name, database_oid, kind)
#
#
# def search_relation_by_db_oid(relation_name, database_oid, kind):
#     results = CATALOG_ANDB_CLASS.search(
#         lambda r: r.database_oid == database_oid and r.name == relation_name and r.kind == kind
#     )
#     if len(results) != 1:
#         raise RollbackError('Not found the table.')
#
#     return results[0].oid


def open_relation(oid, lock_mode=rlock.ACCESS_SHARE_LOCK):
    if oid in __relcache:
        relation = __relcache[oid]
    else:
        results = CATALOG_ANDB_CLASS.search(lambda r: r.oid == oid)
        if len(results) != 1:
            return None

        class_meta = results[0]
        relation = Relation(
            oid=oid, database_oid=class_meta.database_oid, name=class_meta.name
        )
        relation.attrs = CATALOG_ANDB_ATTRIBUTE.search(lambda r: r.class_oid == oid)
        relation.kind = class_meta.kind
        __relcache[oid] = relation

    # try to get lock
    if lock_mode != rlock.NO_LOCK:
        if rlock.lock_acquire(oid, lock_mode, False, 0) == rlock.LOCK_NOT_AVAILABLE:
            return None

    relation.refcount += 1
    return relation


def close_relation(oid, lock_mode=rlock.ACCESS_SHARE_LOCK):
    relation = __relcache[oid]
    if lock_mode != rlock.NO_LOCK:
        if not rlock.lock_release(relation.oid, lock_mode):
            raise RollbackError('cannot release lock')
    relation.refcount -= 1
    if relation.refcount == 0:
        relation.opened = False
        del __relcache[oid]


def hot_create_table(table_name, fields, database_oid=OID_DATABASE_ANDB):
    #TODO: not supported atomic DDL yet
    if CATALOG_ANDB_CLASS.exist_table(table_name, database_oid):
        raise DDLException('the same name table already exists.')

    oid = CATALOG_ANDB_CLASS.create(name=table_name,
                                    kind=RelationKinds.HEAP_TABLE,
                                    database_oid=database_oid)
    #TODO: fields, schema, data file
    # fields format: (name, type_name, notnull)
    CATALOG_ANDB_ATTRIBUTE.define_table_fields(class_oid=oid, fields=fields)
    file_touch(
        os.path.join(BASE_DIR, str(database_oid), str(oid))
    )
    return oid


def hot_drop_table(table_name, database_oid=OID_DATABASE_ANDB):
    #TODO: refactor the return value
    oid = CATALOG_ANDB_CLASS.get_relation_oid(table_name, database_oid, kind=RelationKinds.HEAP_TABLE)
    if oid == INVALID_OID:
        raise DDLException('not found the table.')

    results = CATALOG_ANDB_INDEX.search(lambda r: r.table_oid == oid)
    if len(results) > 0:
        raise DDLException('there are indexes associated with the table.')

    relation = open_relation(oid, rlock.ACCESS_EXCLUSIVE_LOCK)
    if not relation:
        raise DDLException('cannot drop the table because the table is in use.')
    file_close(relation.fd)
    os.unlink(os.path.join(BASE_DIR, str(database_oid), str(oid)))
    CATALOG_ANDB_ATTRIBUTE.delete(lambda r: r.class_oid == oid)
    CATALOG_ANDB_CLASS.delete(lambda r: r.oid == oid)

    # clean buffer
    global_vars.buffer_manager.evict_relation(relation)
    close_relation(oid, rlock.ACCESS_EXCLUSIVE_LOCK)
    return True


def hot_simple_insert(relation: Relation, python_tuple, lsn=None):
    # no redo and undo log here, only update cached page
    #TODO: find from fsm first
    pageno = relation.last_pageno()
    buffer_page = global_vars.buffer_manager.get_page(relation, pageno)
    xid = global_vars.xact_manager.get_xid()
    wip_lsn = 0  # work in progress LSN
    tuple_bytes = TupleData(python_tuple).to_bytes(relation.attrs)
    tid = buffer_page.page.insert(wip_lsn, tuple_bytes)
    if tid == INVALID_ITEM_ID:
        # maybe the page is full, get next pageno
        buffer_page = global_vars.buffer_manager.get_page(relation, pageno + 1)
        tid = buffer_page.page.insert(wip_lsn, tuple_bytes)
        if tid == INVALID_ITEM_ID:
            # still be error? raise the error
            raise RollbackError('cannot insert the tuple')
    buffer_page.mark_dirty()
    
    # write logs
    undo_record = UndoRecord(xid,
                             UndoOperation.HEAP_INSERT,
                             relation, (pageno, tid),
                             b'')
    redo_record = WALRecord(
        xid, relation.oid, pageno, tid, WALAction.HEAP_INSERT, tuple_bytes
    )
    global_vars.xact_manager.undo_manager.write_record(undo_record)
    global_vars.xact_manager.wal_manager.write_record(redo_record)

    # update page header LSN
    if lsn is None:
        lsn = global_vars.xact_manager.max_lsn()
    buffer_page.page.header.lsn = lsn

    return buffer_page.pageno, tid


def hot_simple_update(relation: Relation, pageno, tid, python_tuple):
    # use a same LSN
    lsn = global_vars.xact_manager.max_lsn()
    if hot_simple_delete(relation, pageno, tid, lsn=lsn):
        return hot_simple_insert(relation, python_tuple, lsn=lsn)


def hot_simple_delete(relation: Relation, pageno, tid, lsn=None):
    #TODO: update fsm? or only reorganize?
    buffer_page = global_vars.buffer_manager.get_page(relation, pageno)
    xid = global_vars.xact_manager.get_xid()
    old_tuple_bytes = buffer_page.page.select(tid)
    if old_tuple_bytes == INVALID_BYTES:
        return False

    wip_lsn = 0  # work in progress LSN
    success = buffer_page.page.delete_inplace(wip_lsn, tid)
    if success:
        buffer_page.mark_dirty()
        # write logs
        undo_record = UndoRecord(xid,
                                UndoOperation.HEAP_DELETE,
                                relation, (pageno, tid),
                                old_tuple_bytes)
        redo_record = WALRecord(
            xid, relation.oid, pageno, tid, WALAction.HEAP_DELETE, b''
        )
        global_vars.xact_manager.undo_manager.write_record(undo_record)
        global_vars.xact_manager.wal_manager.write_record(redo_record)

        # update page header LSN
        if lsn is None:
            lsn = global_vars.xact_manager.max_lsn()
        buffer_page.page.header.lsn = lsn
    return success

def hot_batch_delete(relation: Relation, pageno, tid_list):
    # using mark-and-vacuum to delete
    buffer_page = global_vars.buffer_manager.get_page(relation, pageno)
    xid = global_vars.xact_manager.get_xid()
    array_of_old_tuple_bytes = [buffer_page.page.select(tid) for tid in tid_list]
    if any(old_tuple_bytes == INVALID_BYTES for old_tuple_bytes in array_of_old_tuple_bytes):
        return False

    wip_lsn = 0  # work in progress LSN
    for tid in tid_list:
        success = buffer_page.page.delete(wip_lsn, tid)
        if not success:
            return False
    if success:
        buffer_page.mark_dirty()
        # write logs
        undo_record = UndoRecord(xid,
                                UndoOperation.HEAP_BATCH_DELETE,
                                relation, (pageno, tid_list),
                                easy_tuple_serialize(array_of_old_tuple_bytes))
        redo_record = WALRecord(
            xid, relation.oid, pageno, 0, WALAction.HEAP_BATCH_DELETE, easy_tuple_serialize(tid_list)
        )
        global_vars.xact_manager.undo_manager.write_record(undo_record)
        global_vars.xact_manager.wal_manager.write_record(redo_record)

        # update page header LSN
        lsn = global_vars.xact_manager.max_lsn()
        buffer_page.page.header.lsn = lsn
    return success


def hot_simple_select(relation: Relation, pageno, tid):
    buffer_page = global_vars.buffer_manager.get_page(relation, pageno)
    data = buffer_page.page.select(tid)
    if data == INVALID_BYTES:
        return ()
    return TupleData.from_bytes(data, relation.attrs).python_tuple

def hot_simple_select_all(relation: Relation):
    for pageno in range(relation.last_pageno() + 1):
        buffer_page = global_vars.buffer_manager.get_page(relation, pageno)
        for tid in range(buffer_page.page.item_count):
            tuple_ = hot_simple_select(relation, pageno, tid)
            if tuple_ != ():
                yield tuple_

def bt_create_index(index_name, table_name, fields, database_oid=OID_DATABASE_ANDB):
    table_oid = CATALOG_ANDB_CLASS.get_relation_oid(table_name, database_oid, kind=RelationKinds.HEAP_TABLE)
    # only get index columns

    attr_form_array = CATALOG_ANDB_ATTRIBUTE.search(lambda r: r.class_oid == table_oid)
    index_attr_form_array = []
    for field in fields:
        for attr in attr_form_array:
            if attr.name == field:
                index_attr_form_array.append(attr)
    if len(index_attr_form_array) != len(fields):
        raise RollbackError('the index exists invalid field.')
    return bt_create_index_internal(index_name, table_oid, attr_form_array, index_attr_form_array,
                                    database_oid=OID_DATABASE_ANDB)


def bt_create_index_internal(index_name, table_oid, attr_form_array, index_attr_form_array,
                             database_oid=OID_DATABASE_ANDB):
    table_relation = open_relation(table_oid, lock_mode=rlock.SHARE_LOCK)
    if not table_relation:
        raise DDLException('cannot get the table.')

    # generate final index file
    index_oid = CATALOG_ANDB_CLASS.create(index_name, RelationKinds.BTREE_INDEX, database_oid)
    CATALOG_ANDB_INDEX.define_index_fields(name=index_name, index_oid=index_oid,
                                           table_oid=table_oid, table_attr_forms=index_attr_form_array)

    tree = BPlusTree()
    #TODO: fix this lsn
    lsn = global_vars.xact_manager.max_lsn()
    last_pageno = table_relation.last_pageno()
    # iteration includes the last pageno
    for pageno in range(0, last_pageno + 1):
        buffer_page = global_vars.buffer_manager.get_page(table_relation, pageno)
        global_vars.buffer_manager.pin_page(buffer_page)
        hot_page = buffer_page.page
        for idx in range(len(hot_page.item_ids)):
            tuple_data = hot_page.select(idx)
            if tuple_data == INVALID_BYTES:
                continue
            heap_tuple = TupleData.from_bytes(tuple_data, attr_form_array).python_tuple
            key_tuple = tuple(heap_tuple[attr.num] for attr in index_attr_form_array)
            key_data = TupleData(python_tuple=key_tuple).to_bytes(index_attr_form_array)
            tuple_pointer = TuplePointer(pageno, idx)
            tree.insert(lsn, key_data, tuple_pointer)
        global_vars.buffer_manager.unpin_page(buffer_page)

    close_relation(table_oid, lock_mode=rlock.SHARE_LOCK)
    fd = file_open(os.path.join(BASE_DIR, str(database_oid), str(index_oid)),
                   flags=os.O_RDWR | os.O_CREAT)
    file_write(fd, data=tree.serialize(), sync=True)
    return index_oid


def bt_drop_index(index_name, database_oid=OID_DATABASE_ANDB):
    results = CATALOG_ANDB_CLASS.search(lambda r: r.name == index_name and r.database_oid == database_oid)
    if len(results) == 0:
        raise DDLException('Not found the index name %s.' % index_name)

    class_form = results[0]
    index_oid = class_form.oid
    relation = open_relation(index_oid, rlock.ACCESS_EXCLUSIVE_LOCK)
    if not relation:
        raise DDLException('cannot drop the index because the index is in use.')

    CATALOG_ANDB_CLASS.delete(lambda r: r.name == index_name)
    CATALOG_ANDB_INDEX.delete(lambda r: r.oid == class_form.oid)

    file_remove(relation.fd)
    global_vars.buffer_manager.evict_relation(relation)
    close_relation(index_oid, rlock.ACCESS_EXCLUSIVE_LOCK)


def _bt_key_tuple_to_data(key_tuple, index_attr_form_array):
    key_data = TupleData(python_tuple=key_tuple).to_bytes(index_attr_form_array)
    return key_data


def _bt_data_to_key_tuple(data, index_attr_form_array):
    key_tuple = TupleData.from_bytes(data, index_attr_form_array).python_tuple
    return key_tuple


def bt_simple_insert(relation: Relation, key, tuple_pointer):
    tree = BufferedBPTree(relation)
    xid = global_vars.xact_manager.get_xid()
    attrs = CATALOG_ANDB_INDEX.get_attr_form_array(relation.oid)
    key_data = _bt_key_tuple_to_data(key, attrs)

    global_vars.xact_manager.wal_manager.write_record(
        WALRecord(xid, relation.oid, 0, 0, WALAction.BTREE_INSERT, easy_tuple_serialize((key_data, tuple_pointer.to_bytes()))))
    # for delete (the reverse of insert), we need to know the key and the specific value
    global_vars.xact_manager.undo_manager.write_record(
        UndoRecord(xid, UndoOperation.BTREE_INSERT, relation, (key_data, tuple_pointer),
                    b''))

    lsn = global_vars.xact_manager.max_lsn()
    tree.insert(lsn, key_data, tuple_pointer)




def bt_update(relation: Relation, key, tuple_pointer):
    tree = BufferedBPTree(relation)
    xid = global_vars.xact_manager.get_xid()
    attrs = CATALOG_ANDB_INDEX.get_attr_form_array(relation.oid)
    key_data = _bt_key_tuple_to_data(key, attrs)
    global_vars.xact_manager.wal_manager.write_record(
        WALRecord(xid, relation.oid, 0, 0, WALAction.BTREE_UPDATE, 
                  easy_tuple_serialize((key_data, tuple_pointer.to_bytes()))))
    # for update, we need to know the key and value, and let
    # the location be the tuple of key and value
    # the old value is a list of tuple pointer
    old_tuple_pointer = tree.search(key_data)
    global_vars.xact_manager.undo_manager.write_record(
        UndoRecord(xid, UndoOperation.BTREE_UPDATE, relation, (key_data, old_tuple_pointer),
                    b''))

    lsn = global_vars.xact_manager.max_lsn()
    tree.delete(lsn, key_data)
    tree.insert(lsn, key_data, tuple_pointer)


def bt_delete(relation: Relation, key):
    tree = BufferedBPTree(relation)
    xid = global_vars.xact_manager.get_xid()
    attrs = CATALOG_ANDB_INDEX.get_attr_form_array(relation.oid)
    key_data = _bt_key_tuple_to_data(key, attrs)
    global_vars.xact_manager.wal_manager.write_record(
        WALRecord(xid, relation.oid, 0, 0, WALAction.BTREE_DELETE, key_data))
    # for insert (the reverse of delete), we need to know the key and values
    old_tuple_pointer = tree.search(key_data)
    global_vars.xact_manager.undo_manager.write_record(
        UndoRecord(xid, UndoOperation.BTREE_DELETE, relation, (key_data, old_tuple_pointer),
                    b''))

    lsn = global_vars.xact_manager.max_lsn()

    tree.delete(lsn, key_data)


def bt_search(relation: Relation, key):
    """Allow leftmost prefix rule"""
    tree = BufferedBPTree(relation)
    attrs = CATALOG_ANDB_INDEX.get_attr_form_array(relation.oid)
    assert len(key) <= len(attrs)
    key_data = _bt_key_tuple_to_data(key, attrs[:len(key)])
    results = tree.search(key_data)
    return results


def bt_search_range(relation: Relation, start_key, end_key):
    """Allow leftmost prefix rule"""
    tree = BufferedBPTree(relation)
    attrs = CATALOG_ANDB_INDEX.get_attr_form_array(relation.oid)
    assert len(start_key) <= len(attrs)
    assert len(end_key) <= len(attrs)

    start_key_data = _bt_key_tuple_to_data(start_key, attrs[:len(attrs)])
    end_key_data = _bt_key_tuple_to_data(end_key, attrs[:len(attrs)])
    results = tree.search_range(start_key_data, end_key_data)
    return results


def bt_scan_all_keys(relation: Relation):
    tree = BufferedBPTree(relation)
    attrs = CATALOG_ANDB_INDEX.get_attr_form_array(relation.oid)
    for key_data in tree.all_keys():
        yield _bt_data_to_key_tuple(key_data, attrs)

