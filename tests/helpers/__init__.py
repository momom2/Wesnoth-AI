"""Shared test fixtures: code that more than one test file uses.

Import them as `helpers.<module>` (tests/ is on sys.path under pytest and
for a test file run as a script); `tests.helpers.<module>` would load a
second copy of the same file. A test file imports its fixtures from here,
never from another test file, so deleting or renaming a test file breaks
nothing else.
"""
