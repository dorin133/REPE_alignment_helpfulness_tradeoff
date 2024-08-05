import traceback
import signal

# common imports
imports = """
from numpy import concatenate, product, multiply
import json
import re
import numpy as np
import statistics
from statistics import mean
from typing import List, Optional, Tuple, Any
import statistics as stats
import hashlib

"""


def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out!")


# Set the timeout handler for the SIGALRM signal
signal.signal(signal.SIGALRM, timeout_handler)


def test_humaneval_function(humaneval_entry: dict, model_output: str):
    """
    Tests a generated function from the HumanEval dataset.

    :param humaneval_entry: A dictionary containing 'prompt', 'entry_point', and 'test' keys from the HumanEval dataset.
    :param model_output: The code generated by the model as a string.
    :return: True if all test cases pass, False otherwise.
    """
    # dummy_out = DummyOutput()
    # sys.stdout = dummy_out
    def execute_with_timeout(code, timeout_duration, local_dict):
        try:
            # Set the alarm for the timeout duration
            signal.alarm(timeout_duration)

            # Execute the code
            exec(code, local_dict)
        finally:
            signal.alarm(0)

    try:
        local_dict = {}
        # Define the function in the local scope
        execute_with_timeout(imports + model_output, 5, local_dict)
        execute_with_timeout(humaneval_entry['test'], 5, local_dict)

        # Extract the entry point function name
        entry_point = humaneval_entry['entry_point']

        execute_with_timeout(f"check({entry_point})", 5, local_dict)

        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        return False
