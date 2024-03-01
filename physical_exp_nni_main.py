import signal

from nni.experiment import Experiment

import shutil
import os

if os.path.exists("output"):
    shutil.rmtree("output")

# Define search space
search_space = {
    "ms_j": {"_type": "uniform", "_value": [0.05, 0.95]},
    "wall_id": {"_type": "choice", "_value": [0, 1, 2]},
}

# Configure experiment
experiment = Experiment("local")
experiment.config.trial_command = "python physical_exp_nni_model.py"
experiment.config.trial_code_directory = "."
experiment.config.search_space = search_space
experiment.config.max_trial_number = 500
experiment.config.trial_concurrency = 1
experiment.config.tuner.name = "Anneal" # Anneal, Evolution, TPE
experiment.config.tuner.class_args["optimize_mode"] = "minimize"
# experiment.config.tuner.class_args["population_size"] = 16

# Run it!
experiment.run(port=8848)

print("Experiment is running. Press Ctrl-C to quit.")
signal.pause()
