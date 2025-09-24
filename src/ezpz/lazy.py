import importlib.util
import sys


class MissingModuleProxy:
    """Lightweight placeholder returned when an optional dependency is absent."""

    def __init__(self, name: str) -> None:
        self.__name__ = name

    def __getattr__(self, item):
        raise ModuleNotFoundError(
            f"Optional dependency '{self.__name__}' is not available; "
            f"tried to access attribute '{item}'."
        )

    def __repr__(self) -> str:  # pragma: no cover - representational helper
        return f"<MissingModuleProxy name='{self.__name__}'>"


def lazy_import(name: str):
    """
    Lazy import a module.
    """
    try:
        return sys.modules[name]
    except KeyError:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return MissingModuleProxy(name)
        # assert spec is not None, f"Module {name} not found"
        module = importlib.util.module_from_spec(spec)
        assert module is not None, f"Module {name} not found"
        assert spec.loader is not None, f"Module {name} not found"
        loader = importlib.util.LazyLoader(spec.loader)
        # make module with proper locking and get it inserted into sys.modules
        loader.exec_module(module)
        return module
