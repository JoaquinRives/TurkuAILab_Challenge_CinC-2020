import sys


class Logger(object):
    """ Redirects the stdout to both, file and terminal """

    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# sys.stdout = Logger('logfile.log')
