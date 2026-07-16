import logging


def set_verbose(level: str = "INFO") -> None:
    """
    Set logging level for the tpcav_package (and root logger).
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(lvl)
    logging.getLogger("tpcav").setLevel(lvl)
