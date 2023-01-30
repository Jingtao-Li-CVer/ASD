import logging
import os
import datetime

def logger_config(log_path):
    currtime = datetime.datetime.now()
    log_path = log_path.replace(os.path.splitext(log_path)[0], os.path.splitext(log_path)[0] + "-" + str(currtime.month) + "-" + str(currtime.day) + "-" + str(currtime.hour) + "-" + str(currtime.minute))

    
    if os.path.exists(log_path):
        os.remove(log_path)
    
    os.mknod(log_path)
    (logging_name, ext) = os.path.splitext(log_path)
    logger = logging.getLogger(logging_name)
  
    logger.setLevel(level=logging.DEBUG)

    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger