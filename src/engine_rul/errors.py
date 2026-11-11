"""Project exceptions (ERR-01). The CLI maps each to an exit code."""


class EngineRULError(Exception):
    exit_code = 1


class ConfigError(EngineRULError):
    exit_code = 1


class DataValidationError(EngineRULError):
    exit_code = 2


class ChecksumError(DataValidationError):
    exit_code = 2


class TestSetLockError(EngineRULError):
    __test__ = False  # not a pytest test class
    exit_code = 3
