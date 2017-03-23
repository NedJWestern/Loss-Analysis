class extwrapper(object):

    def __init__(self, loader_file_ext):
        self.loader_file_ext = loader_file_ext

    def __call__(self, f):
        def wrapped_f(fname=None):
            if fname is None:
                ret = self.loader_file_ext[f.__name__]
            else:
                ret = f(fname)
            return ret

        return wrapped_f
