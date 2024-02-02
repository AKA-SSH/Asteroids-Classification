# library
import os
import logging
from datetime import datetime

# creating log directory
log_main_path= os.path.join(os.getcwd(), 'logs')
os.makedirs(log_main_path, exist_ok= True)

# naming the log entry
log_date_path= os.path.join(log_main_path, datetime.now().strftime('%Y-%m-%d'))
os.makedirs(log_date_path, exist_ok= True)
log_time_file= f"{datetime.now().strftime('%H-%M-%S')}.log"
log_file_path= os.path.join(log_date_path, log_time_file)

# setting log level
log_level= logging.INFO

# log message
logging.basicConfig(filename= log_file_path,
                    format= '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
                    level= log_level)