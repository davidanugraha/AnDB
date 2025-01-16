from andb.entrance import execute_simple_query
from .conftest import setup_and_teardown, assert_equal_tuples

def test_inner_join(setup_and_teardown):
    query1 = "SELECT title, genre \
        FROM TABULAR(PROMPT('title of the book') AS title text FROM FILE('books.txt')) books JOIN \
        TABULAR(PROMPT('genre') AS genre text FROM FILE('genres.txt')) genres \
        ON SEM_MATCH('The book {books.title} is of genre {genres.genre}')"

    result1_first_run = execute_simple_query(query1).tuples
    result1_second_run = execute_simple_query(query1).tuples
    
    assert(len(result1_first_run) >= 20) # At least, but only check length since it's not deterministic
    for tup in result1_first_run:
        assert(len(tup) == 2)
    
    assert_equal_tuples(result1_first_run, result1_second_run)
    
def test_inner_join_with_where(setup_and_teardown):
    query1 = "SELECT title, genre \
        FROM TABULAR(PROMPT('title of the book') AS title text FROM FILE('books.txt')) books JOIN \
        TABULAR(PROMPT('genre') AS genre text FROM FILE('genres.txt')) genres \
        ON SEM_MATCH('The book {books.title} is of genre {genres.genre}') \
        WHERE SEM_MATCH('The genre {genres.genre} is scary')"

    result1_first_run = execute_simple_query(query1).tuples
    result1_second_run = execute_simple_query(query1).tuples
    
    assert(len(result1_first_run) >= 5) # At least, but only check length since it's not deterministic
    for tup in result1_first_run:
        assert(len(tup) == 2)
    
    assert_equal_tuples(result1_first_run, result1_second_run)

