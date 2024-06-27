import logging
from logging.config import dictConfig

LOGGING_HANDLER_REGISTERED = False

# Logging configuration setup
def setup_logging():
    """
    Sets up the logger such that it can be used throughout the rest of the code by importing the "logging" module.
    :param logfile:
    :return:
    """

    global LOGGING_HANDLER_REGISTERED
    if LOGGING_HANDLER_REGISTERED:
        logging.info('Logging handler already registered. Skipping adding another handler')
        return
        
    LOGGING_LEVEL = logging.DEBUG
    LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': LOGGING_FORMAT,
            'style': '%',
        }},
        'handlers': {'default': {
            'level': LOGGING_LEVEL,
            'formatter': 'default',
            'class': 'logging.StreamHandler',
        }},
        'root': {
            'handlers': ['default'],
            'level': LOGGING_LEVEL,
            'propagate': True,
        }
    })
    

    
    LOGGING_HANDLER_REGISTERED = True


if __name__=='__main__':
    setup_logging()
    logging.info('test')