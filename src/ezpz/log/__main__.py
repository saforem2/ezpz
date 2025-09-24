from ezpz.log import get_logger

log = get_logger(__name__)


def test_log():
    import json

    import numpy as np
    import pandas as pd

    log.debug("Debug")
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")
    log.info("Long lines are automatically wrapped by the terminal!")
    log.info("[start]" + 250 * "-" + "[end]")
    x = np.arange(16).reshape(4, 4)
    data = {
        "a": 1,
        "b": 2,
        "c": 3.0,
    }

    df = pd.DataFrame().from_dict(data, orient="index")
    log.info(f"json:\n{json.dumps(data, indent=4)}")
    log.info(f"array:\n{np.array2string(x)}")
    log.info(f"dataframe:\n{df.T}")


if __name__ == "__main__":
    test_log()
