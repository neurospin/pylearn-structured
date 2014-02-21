# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.tests.tests` module contains basic functionality for unit
testing. It also has the ability to run all unit tests.

Created on Wed Feb 19 14:55:58 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from nose.tools import nottest
import unittest
import abc
import os

__all__ = ["TestCase", "test_all"]


class TestCase(unittest.TestCase):
    """Unit test base class.

    Example
    -------
    Add a test method:

        def test_1(self):
            assert True
    """
    __metaclass__ = abc.ABCMeta

    def setup(self):
        """This method is run before each unit test.

        Specialise if you need to setup something before each test method is
        run.
        """
        pass

    def teardown(self):
        """This method is run after each unit test.

        Specialise if you need to tear something down after each test method
        is run.
        """
        pass

    @classmethod
    def setup_class(cls):
        """This method is run before any other methods in this class.

        Specialise if you need to setup something before the test commences.
        """
        pass

    @classmethod
    def teardown_class(cls):
        """This method is run after all other methods in this class.

        Specialise if you need to tear something down after all these unit
        tests are done.
        """
        pass


@nottest
def test_all():

    # Find parsimony directory.
    # TODO: Is there a better way to do this?
    testdir = os.path.dirname(__file__)
    if len(testdir) == 0:
        testdir = ".."
    elif testdir[-1] == '/':
        testdir += ".."
    else:
        testdir += "/.."

    # --exclude='^playhouse\\.py$'
    exec_string = "nosetests --with-doctest --doctest-tests " + \
                  "--with-coverage -vv -w %s" \
                  % (testdir,)

    print "Running: " + exec_string
    os.system(exec_string)

if __name__ == "__main__":
    test_all()