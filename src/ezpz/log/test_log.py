from ezpz.log import get_logger

log = get_logger(__name__)


def test_log():
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")
    log.info("Long lines are automatically wrapped by the terminal")
    log.info(250 * "-")


if __name__ == "__main__":
    test_log()
