import logging
applogger = logging.getLogger('myapp')
hdlr = logging.FileHandler('/var/tmp/myapp.log')
formatter = logging.Formatter('%(thread)d %(threadName)s %(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
applogger.addHandler(hdlr)
applogger.setLevel(logging.DEBUG)
