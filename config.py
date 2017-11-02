#!/usr/bin/env python
import logging
import os

import pytesseract
import pandas as pd

pd.set_option('display.width', 200)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s]:{} %(levelname)s %(message)s'.format(os.getpid()))
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

DATA_ROOT = '../../data/'