import logging
import os
from time import strftime, gmtime

from data import LOGGING_DIR


def setup_logger():
    # Setup Logging
    timestamp = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    log_name = '{}_{}'.format('MULTI-EURLEX', timestamp)

    # Clean loggers for GCS
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(LOGGING_DIR, log_name + '.txt'),
                        filemode='a')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
