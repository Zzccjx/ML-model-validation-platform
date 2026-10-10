import logging
import sys

class SensitiveFilter(logging.Filter):
    def filter(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for word in ['password', 'token', 'secret']:
                if word in record.msg.lower():
                    record.msg = record.msg.replace(word, '***')
        return True

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger('ai_validation')
    logger.setLevel(log_level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        sensitive_filter = SensitiveFilter()
        logger.addFilter(sensitive_filter)
        
    return logger

logger = setup_logging()
