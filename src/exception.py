import sys
from src.logger import logger

def error_message_detail(error, error_detail: sys):
    """
    Extracts the exact file name, line number and error message from the traceback.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        # Generate the detailed error message
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

        # Automatically log the error when the exception is raised
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message