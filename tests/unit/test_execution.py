from andb.entrance import execute_simple_query


def test_execute_simple_query():
    execute_simple_query('create table t1 (a int not null, b text)')
    execute_simple_query("insert into t1 values (1, 'aaa')")
    execute_simple_query("insert into t1 values (2, 'bbb')")
    execute_simple_query("insert into t1 values (3, null)")
    execute_simple_query("insert into t1 values (4, 'ccc')")
    execute_simple_query("select * from t1 order by a, b")
    execute_simple_query("select * from t1 order by a, b DESC")
    execute_simple_query("delete from t1 where a = 4")
    execute_simple_query("delete from t1")
    execute_simple_query("insert into t1 values (1, 'aaa')")
    execute_simple_query("insert into t1 values (2, 'bbb')")
    execute_simple_query("insert into t1 values (3, null)")
    execute_simple_query("insert into t1 values (4, 'ccc')")
    execute_simple_query("update t1 set a = 5 where b = 'ccc'")
    execute_simple_query("insert into t1 values (4, 'ccc')")
    execute_simple_query("insert into t1 values (4, 'ccc')")
    execute_simple_query("select * from t1;")
    execute_simple_query("select * from t1 where a = 1;")
    execute_simple_query("select * from t1 where a > 2;")
    execute_simple_query("select b from t1 where a > 2;")
    execute_simple_query("select a, count(a) from t1 where a > 2 group by a;")
    execute_simple_query("select a, count(a) from t1 where a > 2 group by a having a > 3;")

    execute_simple_query("create table t2 (a int, city text)")
    execute_simple_query("insert into t2 values (1, 'beijing')")
    execute_simple_query("insert into t2 values (2, 'shanghai')")
    execute_simple_query("insert into t2 values (3, 'guangdong')")
    execute_simple_query("insert into t2 values (4, 'shenzhen')")
    execute_simple_query("select t1.a, t2.city from t1, t2")

    execute_simple_query("explain select t1.a, city from t1, t2 where t1.a = t2.a")
    execute_simple_query("select t1.a, t2.city from t1, t2 where t1.a = t2.a")

    execute_simple_query('create index idx1 on t1 (a)')
    execute_simple_query('select a from t1')
    execute_simple_query("select t1.a, city from t1, t2 where t1.a = t2.a;")
    execute_simple_query("explain select t2.a, city from t1, t2 where t1.a = t2.a;")
