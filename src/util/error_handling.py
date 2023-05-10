import sys
import traceback


class LogExceptionAndSkip:
    def __init__(self, name, cleanup_fn=None, file=None):
        self.name = name
        self.file = file
        self.cleanup_fn = cleanup_fn

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            file=self.file or sys.stderr

            print(f"Encountered the following while trying to {self.name}:", file=file)
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=file)
            if self.cleanup_fn is not None:
                self.cleanup_fn()
        return True
