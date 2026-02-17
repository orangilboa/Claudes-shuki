# No print statements in production code

Never add print() statements, console.log(), or similar debug output to
production code files. If logging is needed, use the project's existing
logger (e.g. logging.getLogger(__name__).info(...) in Python).
