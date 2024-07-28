import contextlib
import io

def suppress_output(func):
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)
    return wrapper
