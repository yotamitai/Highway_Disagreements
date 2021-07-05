import logging
from datetime import datetime
from os import makedirs
from os.path import abspath, exists, basename, join


def get_logging(args):
    if not exists(abspath('logs')):
        makedirs('logs')
    name = '-'.join([basename(args.a1_name), basename(args.a2_name)])
    file_name = '_'.join([name, datetime.now().strftime("%d-%m %H:%M:%S").replace(' ', '_')])
    log_name = join('logs', file_name)
    args.output = join('results', file_name)
    logging.basicConfig(filename=log_name + '.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    log(f'Comparing Agents: {name}', args.verbose)
    log(f'Disagreement importance by: {args.importance_type}', args.verbose)
    return name, file_name


def log(msg, verbose=False):
    if verbose: print(msg)
    logging.info(msg)