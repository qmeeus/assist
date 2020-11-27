import os, os.path as p
import logzero
import logging
from logzero import logger
from pathlib import Path


LOGLEVEL = getattr(logging, os.environ.get("LOGLEVEL", "INFO"))

logdir = (Path(__file__).parents[2]/"exp/logs").absolute()
os.makedirs(logdir, exist_ok=True)
logfile = logdir/"logs.txt"

# Setup rotating logfile with 3 rotations, each with a maximum filesize of 1MB:
logzero.logfile(logfile, maxBytes=1e6, backupCount=3)
logzero.loglevel(LOGLEVEL)
