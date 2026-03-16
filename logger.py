""" 
Logger Utility 
 
Provides a centralized logger configuration for the 
SourceBytes Agent Engine. 
""" 
 
import logging 
import sys 
from settings import LOG_LEVEL 
 
 
# ------------------------------------------------------- 
# Create Logger 
# ------------------------------------------------------- 
 
def get_logger(name: str) -> logging.Logger: 
    """ 
    Create and return a configured logger instance. 
    """ 
 
    logger = logging.getLogger(name) 
 
    if logger.handlers: 
        return logger 
 
    logger.setLevel(LOG_LEVEL) 
 
    formatter = logging.Formatter( 
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S" 
    ) 
 
    handler = logging.StreamHandler(sys.stdout) 
    handler.setFormatter(formatter) 
 
    logger.addHandler(handler) 
 
    return logger 