import sys
import traceback


class LogExceptionAndSkip:
    def __init__(self, name, cleanup_fn=None):
        self.name = name
        self.cleanup_fn = cleanup_fn

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print(f"Encountered the following while trying to {self.name}:")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
        if self.cleanup_fn is not None:
            self.cleanup_fn()
        return True
