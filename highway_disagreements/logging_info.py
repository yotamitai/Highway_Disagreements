import json
import logging
from datetime import datetime
from os import makedirs
from os.path import abspath, exists, basename, join
from pathlib import Path

from highway_disagreements.utils import make_clean_dirs


def getmylogger(name, path):
    file_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s -- %(message)s')
    file_handler = logging.FileHandler(join(path,"logfile.log"))
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger = logging.getLogger(name)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def get_logging(args):
    if not exists(abspath('logs')):
        makedirs('logs')
    name = '-'.join([basename(args.a1_name), basename(args.a2_name)])
    file_name = '_'.join([datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_'), name])
    args.output = join(args.results_dir, file_name)
    make_clean_dirs(args.output)
    with Path(join(args.output, 'metadata.json')).open('w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    logger = getmylogger('ComparisonLogger', args.output)
    log(logger, f'Comparing Agents: {name}', args.verbose)
    log(logger, f'Disagreement importance by: {args.importance}', args.verbose)
    return name, logger


def log(logger, msg, verbose=False):
    if verbose: print(msg)
    logger.info(msg)
