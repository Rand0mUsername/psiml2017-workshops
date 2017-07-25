import os
from time import time


class ProjectPath:
    base = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, logdir):
        self.logdir = logdir

        from time import localtime, strftime
        self.timestamp = strftime("%B_%d__%H:%M", localtime())

        self.model_path = os.path.join(ProjectPath.base, self.logdir, self.timestamp)
        # this is crucial for multiple runs!
        # tensorboard crawls the log folder and 1 Run = 1 subfolder with tfevents data
        # that's why we write logs to a different subfolder in every run


class Timer:
    def __init__(self):
        self.curr_time = time()

    def time(self):
        diff = time() - self.curr_time
        self.curr_time = time()
        return diff
