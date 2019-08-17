from collections import defaultdict
import json
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from updown.types import Prediction


class NocapsEvaluator(object):
    r"""
    A utility class to submit model predictions on nocaps splits to EvalAI, and retrieve model
    performance based on captioning metrics (such as CIDEr, SPICE).

    Extended Summary
    ----------------
    This class and the training script together serve as a working example for "EvalAI in the
    loop", showing how evaluation can be done remotely on privately held splits. Annotations
    (captions) and evaluation-specific tools (e.g. `coco-caption <https://www.github.com/tylin/coco-caption>`_)
    are not required locally. This enables users to select best checkpoint, perform early
    stopping, learning rate scheduling based on a metric, etc. without actually doing evaluation.

    Parameters
    ----------
    phase: str, optional (default = "val")
        Which phase to evaluate on. One of "val" or "test".

    Notes
    -----
    This class can be used for retrieving metrics on both, val and test splits. However, we
    recommend to avoid using it for test split (at least during training). Number of allowed
    submissions to test split on EvalAI are very less, and can exhaust in a few iterations! However,
    the number of submissions to val split are practically infinite.
    """

    def __init__(self, phase: str = "val"):

        # Constants specific to EvalAI.
        self._challenge_id = 355
        self._phase_id = 742 if phase == "val" else 743

    def evaluate(
        self, predictions: List[Prediction], iteration: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        r"""
        Take the model predictions (in COCO format), submit them to EvalAI, and retrieve model
        performance based on captioning metrics.

        Parameters
        ----------
        predictions: List[Prediction]
            Model predictions in COCO format. They are a list of dicts with keys
            ``{"image_id": int, "caption": str}``.
        iteration: int, optional (default = None)
            Training iteration where the checkpoint was evaluated.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Model performance based on all captioning metrics. Nested dict structure::

                {
                    "B1": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-1
                    "B2": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-2
                    "B3": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-3
                    "B4": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-4
                    "METEOR": {"in-domain", "near-domain", "out-domain", "entire"},
                    "ROUGE-L": {"in-domain", "near-domain", "out-domain", "entire"},
                    "CIDEr": {"in-domain", "near-domain", "out-domain", "entire"},
                    "SPICE": {"in-domain", "near-domain", "out-domain", "entire"},
                }

        """
        # Save predictions as a json file first.
        _, predictions_filename = tempfile.mkstemp(suffix=".json", text=True)
        with open(predictions_filename, "w") as f:
            json.dump(predictions, f)

        submission_command = (
            f"evalai challenge {self._challenge_id} phase {self._phase_id} "
            f"submit --file {predictions_filename}"
        )

        submission_command_subprocess = subprocess.Popen(
            submission_command.split(),
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # This terminal output will have submission ID we need to check.
        submission_command_stdout = submission_command_subprocess.communicate(input=b"N\n")[
            0
        ].decode("utf-8")

        submission_id_regex = re.search("evalai submission ([0-9]+)", submission_command_stdout)
        try:
            # Get an integer submission ID (as a string).
            submission_id = submission_id_regex.group(0).split()[-1]  # type: ignore
        except:
            # Very unlikely, but submission may fail because of some glitch. Retry for that.
            return self.evaluate(predictions)

        if iteration is not None:
            print(f"Submitted predictions for iteration {iteration}, submission id: {submission_id}.")
        else:
            print(f"Submitted predictions, submission_id: {submission_id}")

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

            # Raise error if it takes more than 5 minutes.
            if num_tries == 30:
                raise ConnectionError("Unable to get results from EvalAI within 5 minutes!")

        # Convert result to json.
        metrics = json.loads(result_stdout, encoding="utf-8")

        # keys: {"in-domain", "near-domain", "out-domain", "entire"}
        # In each of these, keys: {"B1", "B2", "B3", "B4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"}
        metrics = {
            "in-domain": metrics[0]["in-domain"],
            "near-domain": metrics[1]["near-domain"],
            "out-domain": metrics[2]["out-domain"],
            "entire": metrics[3]["entire"],
        }

        # Restructure the metrics dict for better tensorboard logging.
        # keys: {"B1", "B2", "B3", "B4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"}
        # In each of these, keys: keys: {"in-domain", "near-domain", "out-domain", "entire"}
        flipped_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        for key, val in metrics.items():
            for subkey, subval in val.items():
                flipped_metrics[subkey][key] = subval

        return flipped_metrics
