"""Contains the actual entrypoint to the application."""

import sys
import logging


class Application:
    """Represents the federated learning application."""


    def __init__(self) -> None:
        """Initializes a new Application instance."""

        self.logger = logging.getLogger('fl')
        self.logger.setLevel(logging.DEBUG)
        logging_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        console_logging_handler = logging.StreamHandler(sys.stdout)
        console_logging_handler.setLevel(logging.DEBUG)
        console_logging_handler.setFormatter(logging_formatter)
        self.logger.addHandler(console_logging_handler)

    def run(self) -> None:
        """Runs the application."""

        self.logger.info('Hello from a federated World!')
