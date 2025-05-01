import importlib.util
import sys

def lazy_import(name: str):
    """
    Lazy import a module.
    """
    try:
        return sys.modules[name]
    except KeyError:
        spec = importlib.util.find_spec(name)
        assert spec is not None, f"Module {name} not found"
        module = importlib.util.module_from_spec(spec)
        assert module is not None, f"Module {name} not found"
        assert spec.loader is not None, f"Module {name} not found"
        loader = importlib.util.LazyLoader(spec.loader)
        # make module with proper locking and get it inserted into sys.modules
        loader.exec_module(module)
        return module
