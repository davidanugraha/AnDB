import os
import shutil

import pytest

from andb.cmd.setup import initialize_data_dir
from andb.initializer import init_all_database_components

TEST_DATA_DIRECTOR = os.path.realpath('test_data')


def setup():
    if os.path.exists(TEST_DATA_DIRECTOR):
        shutil.rmtree(TEST_DATA_DIRECTOR)
    initialize_data_dir(TEST_DATA_DIRECTOR)

    init_all_database_components()


def teardown():
    shutil.rmtree(TEST_DATA_DIRECTOR, ignore_errors=True)


@pytest.fixture(scope='session', autouse=True)
def run_before_and_after_test_case():
    setup()

    yield

    teardown()
