from collections import defaultdict
import json
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List

from mypy_extensions import TypedDict


Prediction = TypedDict("Prediction", {"image_id": int, "caption": str})


class NocapsEvaluator(object):
    def __init__(self, phase: str = "val"):

        # Constants specific to EvalAI.
        self._challenge_id = 355
        self._phase_id = 742 if phase == "val" else 743

    def evaluate(self, predictions: List[Prediction]):
        # Save predictions as a json file first.
        _, predictions_filename = tempfile.mkstemp(suffix=".json", text=True)
        json.dump(predictions, open(predictions_filename, "w"))

        submission_command = f"evalai challenge {self._challenge_id} phase {self._phase_id} " \
                             f"submit --file {predictions_filename}"

        submission_command_subprocess = subprocess.Popen(
            submission_command.split(),
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # This terminal output will have submission ID we need to check.
        submission_command_stdout = submission_command_subprocess.communicate(input=b"N\n")[0].decode("utf-8")

        submission_id_regex = re.search("evalai submission ([0-9]+)", submission_command_stdout)

        # Get an integer submission ID (as a string).
        submission_id = submission_id_regex.group(0).split()[-1]  # type: ignore

        # Placeholder stdout for a pending submission.
        result_stdout: str = "The Submission is yet to be evaluated."
        num_tries: int = 0

        # Query every 10 seconds for result until it appears.
        while "CIDEr" not in result_stdout:

            time.sleep(10)

            result_stdout = subprocess.check_output(
                ["evalai", "submission", submission_id, "result"]
            ).decode("utf-8")

            num_tries += 1

            # Raise error if it takes more than 10 minutes.
            if num_tries == 60:
                raise ConnectionError("Unable to get results from EvalAI within 10 minutes!")

        # Convert result to json.
        # keys: {"in-domain", "near-domain", "out-domain", "entire"}
        # In each of these, keys: {"B1", "B2", "B3", "B4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"}
        metrics = json.loads(result_stdout, encoding="utf-8")

        # Restructure the metrics dict for better tensorbaord logging.
        metrics = {
            "in-domain": metrics[0]["in-domain"],
            "near-domain": metrics[1]["near-domain"],
            "out-domain": metrics[2]["out-domain"],
            "entire": metrics[3]["entire"],
        }

        flipped_metrics: Dict[str, Any] = defaultdict(dict)
        for key, val in metrics.items():
            for subkey, subval in val.items():
                flipped_metrics[subkey][key] = subval

        # keys: {"B1", "B2", "B3", "B4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"}
        # In each of these, keys: keys: {"in-domain", "near-domain", "out-domain", "entire"}
        metrics = flipped_metrics

        return metrics
