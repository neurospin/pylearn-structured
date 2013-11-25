Architecture
============

- Do we have any architechtural constraints?



General guidelines
==================

- Functions are better if they are short (try to keep them less than 25 lines, and absolutely less than 50).
- Make functions and classes, or code in general, as clear as possible. Use explicit and descriptive names as far as possible.
- Some comments are very important, for example when choices have been made or if something is an opition.
- Use standard comment prefixes, that are easy to search, and which are adequate for: TODO, FIXME, WARNING, ERROR, BUG, XXX, etc.
- Boy scout rule: Leave the campground cleaner than you found it. 
- Write your method from top to down. For example: if a method A use a method B, write A before you write B.
- Class order: top -> down, high level code -> low level code.
- Don't leave prints (writes to stdout).
- Don't optimise prematurely. Make it work well, before you spend time making it fast.



Python code format
==================

- Pep 8. As far as possible.
- Never go beyond 80 characters per line. If you cannot break the line before 80 characters, well then you or someone else have done something wrong.
- Keep modules short and purposeful.
- All the fields of an instance should be explictly declared in the function "__init__" or in methods that have the prefix "_init_". Avoid adding fields run-time.
- Never "import *".
- Do not use relative imports. When we move to Python 3, this will be relevant.
- First import standard python modules, add one empty line and then import other non-standard python modules, add another empty line and then import library modules.
- Use coherent names when importing modules, and use the same thoughout. E.g. "import numpy as np", "import very.long.imported.module.path as path".
- Private fields, classes, functions, etc. use an underscore prefix. Such as "_private_field = 3.141592653589".
- Lists, tuples, modules or datatypes with multiples members should be names in plural.

Docstring format
----------------

- Use PEP 257 convention.
- Here is a useful example using the same style as numpy:

    Testing code block.

You may omit any parts you feel are not relevant for your function.



Commit Message
==============

Git format
----------

- Write proper sentences.
- Be descriptive.
- Use several lines if necessary, wrap at 80 characters.

Possible tag list for the first line
------------------------------------

Should we use this?

- **ENH**: When adding or improving an existing or new class in term of capabilities,
- **COMP**: When fixing a compilation error or warning,
- **DOC**: When starting or improving the class documentation,
- **STYLE**: When enhancing the comments or coding style without any effect on the class behaviour,
- **REFAC**: When refactoring without adding new capabilities,
- **BUG**: When fixing a bug (if the bug is identified in the tracker, please add the reference),
- **INST**: When fixing issues related to installation,
- **PERF**: When improving performance,
- **TEST**: When adding or modifying a test,
- **WRG**: When correcting a warning.
