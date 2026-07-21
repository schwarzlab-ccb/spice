import logging
import os
from datetime import datetime
import yaml

# No SPICE imports here!

# Global logging configuration (set by CLI or defaults)
_LOGGING_CONFIG = {
    'mode': 'terminal',  # 'terminal', 'file', or 'both'
    'log_dir': None,
    'config_name': None,
    'level': 'INFO',
    'file_handler': None,  # Shared file handler for all loggers
    'configured': False
}


def configure_logging(log_mode, log_dir, config_name, level: str = 'INFO'):
    """
    Configure global logging settings for all SPICE loggers.
    Should be called once by CLI before any logging occurs.

    Args:
        log_mode: 'terminal', 'file', or 'both'
        log_dir: Directory for log files
        config_name: Name from config (used in log filename)
        level: Logging level string (e.g., 'INFO', 'DEBUG')
    """
    global _LOGGING_CONFIG

    _LOGGING_CONFIG['mode'] = log_mode
    _LOGGING_CONFIG['log_dir'] = log_dir
    _LOGGING_CONFIG['config_name'] = config_name
    _LOGGING_CONFIG['level'] = level
    _LOGGING_CONFIG['configured'] = True

    # Create shared file handler if needed
    if log_mode in ['file', 'both']:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{config_name}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)

        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('*** %(name)s - %(levelname)s - %(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        _LOGGING_CONFIG['file_handler'] = file_handler


    # Set the level for all active loggers
    logging_level = level.upper()
    logging_values = {
        'SILENT': 'WARNING',
        'VERBOSE': 'DEBUG',
    }
    logging_level = logging_values.get(logging_level, logging_level)
    assert logging_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if 'spice' not in logger_name.lower():
            continue
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging_level)
        # Remove all handlers
        logger.handlers.clear()
        formatter = logging.Formatter('*** %(name)s - %(levelname)s - %(asctime)s - %(message)s')
        # Add terminal handler if needed
        if log_mode in ['terminal', 'both']:
            terminal_handler = logging.StreamHandler()
            terminal_handler.setFormatter(formatter)
            logger.addHandler(terminal_handler)
        # Add file handler if needed and available
        if log_mode in ['file', 'both'] and _LOGGING_CONFIG['file_handler'] is not None:
            logger.addHandler(_LOGGING_CONFIG['file_handler'])
        # Prevent double logging by disabling propagation
        logger.propagate = False


def get_logger(name, spice_prefix=True, load_config=True, config_file=None):
    """
    Get a logger that respects global logging configuration.

    Args:
        name: Logger name
        load_config: Whether to load logging level from config (backward compatibility)
        config_file: Config file to load (backward compatibility)

    Returns:
        Configured logger instance
    """
    global _LOGGING_CONFIG

    # Determine logging level
    if load_config and config_file is None:
        # For installed packages, try to access via importlib.resources
        try:
            import sys
            if sys.version_info >= (3, 9):
                from importlib.resources import files
            else:
                from importlib_resources import files
            default_cfg_content = files('spice').joinpath('objects', 'default_config.yaml').read_text()
            config = yaml.safe_load(default_cfg_content)
            logging_level = config['params']['logging_level']
        except (ImportError, AttributeError, TypeError, FileNotFoundError, KeyError):
            pass

    if _LOGGING_CONFIG['configured'] and _LOGGING_CONFIG.get('level'):
        logging_level = _LOGGING_CONFIG['level']
    else:
        if config_file is not None:
            # config_file should now always be a string path
            with open(config_file, 'rt') as f:
                config = yaml.safe_load(f.read())
            logging_level = config['params']['logging_level']
        else:
            logging_level = "INFO"

    if spice_prefix and not name.startswith('spice.'):
        name = f'spice.{name}'
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    logger.handlers.clear()  # Clear any existing handlers

    formatter = logging.Formatter('*** %(name)s - %(levelname)s - %(asctime)s - %(message)s')

    # If global config is set, use it; otherwise default to terminal
    log_mode = _LOGGING_CONFIG['mode'] if _LOGGING_CONFIG['configured'] else 'terminal'

    # Add terminal handler if needed
    if log_mode in ['terminal', 'both']:
        terminal_handler = logging.StreamHandler()
        terminal_handler.setFormatter(formatter)
        logger.addHandler(terminal_handler)

    # Add file handler if needed and available
    if log_mode in ['file', 'both'] and _LOGGING_CONFIG['file_handler'] is not None:
        logger.addHandler(_LOGGING_CONFIG['file_handler'])

    # Prevent double logging by disabling propagation
    logger.propagate = False

    return logger


def set_logging_level(logger, level):
    logging_values = {
        'silent': 'WARNING',
        'verbose': 'DEBUG',
    }
    level = logging_values.get(level, level.upper())
    assert level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    logger.setLevel(level)


def log_debug(logger, msg):
    if logger.level == logging.DEBUG:
        logger.debug(msg)
