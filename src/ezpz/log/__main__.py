import logging
from ezpz.configs import get_logging_config
from ezpz.log import get_logger, _get_logger

log = get_logger(__name__)
# log = _get_logger(__name__)
# log = logging.getLogger(__name__)

# log.setLevel('DEBUG')
# log_config = logging.config.dictConfig(get_logging_config())
# log = logging.getLogger(__name__)
# log.setLevel('INFO')


def test_log():
    import numpy as np
    import pandas as pd
    log.debug('Debug')
    log.info('info')
    log.warning('warning')
    log.error('error')
    log.critical('critical')
    log.info('Long lines are automatically wrapped by the terminal!')
    log.info(250 * '-')
    x = np.random.rand(100).reshape(10, 10)
    data = {
        'a': 1,
        'b': 2,
        'c': 3.,
    }
    import json
    log.info(f'data: {json.dumps(data, indent=4)}')
    df = pd.DataFrame().from_dict(data, orient='index')
    log.info(f'{x=}')
    log.info(f'{df=}')


if __name__ == '__main__':
    test_log()
