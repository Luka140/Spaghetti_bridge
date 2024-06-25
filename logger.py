import pathlib
from datetime import datetime


def logger(results_directory, filename, **kwargs):
    print("\n\n Logging files \n\n")
    stamped_filename = pathlib.Path(results_directory) / f'{filename}{datetime.now()}.log'
    with open(stamped_filename, 'w') as f:
        for key, value in kwargs.items():
            f.write(f'{key} - {value}\n')
