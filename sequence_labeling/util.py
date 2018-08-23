import os


class SimpleLogger(object):
    def __init__(self, folder=None, logfile='log.txt'):
        if folder is None:
            self.log2file = False
        else:
            self.log2file = True
            self.logfile = os.path.join(folder, logfile)
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(self.logfile, 'w'):
                pass

    def __call__(self, *s):
        print(*s)
        if self.log2file:
            with open(self.logfile, 'a+') as f_log:
                f_log.write(' '.join([str(ss) for ss in s]) + '\n')
