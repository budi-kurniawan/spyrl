from pathlib import Path

def override(parent_class):
    def overrider(method):
        assert(method.__name__ in dir(parent_class))
        return method
    return overrider


def get_project_dir() -> Path:
    return Path(__file__).parent.parent.parent

