import re
import time

import numpy as np
import requests

from biomni.task.base_task import base_task as BaseTask
from biomni.tool.neuroscience import generate_synthetic_spike_train


# --- Task 1: Spike Train Classification ---
class NeuronClassificationTask(BaseTask):
    def __init__(self):
        self.name = "neuron_classification"
        self.description = "Identify if a neuron is 'regular', 'poisson', or 'bursting' based on its spike train."

    def __len__(self):
        return 20  # 20 test cases generated on the fly

    def get_example(self, index):
        # Deterministic random seed for reproducibility based on index
        np.random.seed(index)

        # Rotate through patterns
        patterns = ["regular", "poisson", "bursty"]
        ground_truth = patterns[index % 3]

        # Generate data
        spikes = generate_synthetic_spike_train(duration=4.0, pattern=ground_truth)

        spikes_list = spikes.tolist()

        return {
            "instruction": (
                f"I have a neuron recording with a duration of 4.0 seconds. "
                f"The spike times (in seconds) are: {spikes_list}. "
                f"Analyze the inter-spike intervals and Coefficient of Variation (CV). "
                f"Classify this neuron's firing pattern as 'regular', 'poisson', or 'bursty'."
            ),
            "ground_truth": ground_truth,
        }

    def evaluate(self, prediction, ground_truth):
        # Simple keyword matching
        pred_lower = prediction.lower()

        # Map "bursting" and "bursty" to the same concept
        if ground_truth == "bursty":
            return "burst" in pred_lower
        elif ground_truth == "regular":
            return "regular" in pred_lower or "pacemaker" in pred_lower
        elif ground_truth == "poisson":
            return "poisson" in pred_lower or "irregular" in pred_lower
        return False


# --- Task 2: Calcium Event Detection ---
class CalciumEventTask(BaseTask):
    def __init__(self):
        self.name = "calcium_event_counting"
        self.description = "Count the number of significant calcium events in a fluorescence trace."

    def __len__(self):
        return 10

    def get_example(self, index):
        np.random.seed(index + 100)

        # Generate a synthetic calcium trace
        frames = 300  # 10 seconds @ 30Hz
        trace = np.random.normal(0, 1, frames)  # Baseline noise

        # Inject known events (peaks)
        num_events = np.random.randint(1, 5)
        for _ in range(num_events):
            pos = np.random.randint(20, 280)
            # Add a "calcium transient" (exponential decay)
            trace[pos : pos + 20] += np.exp(-np.arange(0, 2, 0.1)) * 10  # Strong signal (10 sigma)

        return {
            "instruction": (
                f"Here is a raw fluorescence trace (30 Hz framerate) from a single neuron: {trace.tolist()}. "
                f"Detect the number of significant calcium events (peaks > 3 sigma). "
                f"Return ONLY the integer count."
            ),
            "ground_truth": str(num_events),
        }

    def evaluate(self, prediction, ground_truth):
        # Extract the first number found in the prediction
        numbers = re.findall(r"\d+", prediction)
        if not numbers:
            return False

        # We allow a margin of error of +/- 1 event due to threshold sensitivity
        predicted_count = int(numbers[0])
        true_count = int(ground_truth)

        return abs(predicted_count - true_count) <= 0


def get_allen_reference_data(region="VISp", num_cells=5):
    """
    Fetches real cell data from Allen Brain Atlas to use as Ground Truth for evaluation.
    """
    url = "http://api.brain-map.org/api/v2/data/query.json"

    # Query for cells with ephys data in the specified region
    query = (
        "criteria=model::Specimen,"
        "rma::criteria,[is_cell_specimen$eq'true'],"
        f"structure[name$il'*{region}*'],"
        "ephys_features"
    )

    try:
        # We use a timeout to prevent the eval from hanging if the API is down
        response = requests.get(url, params={"q": query}, timeout=10)
        data = response.json()

        if data["success"] and len(data["msg"]) > 0:
            # Return a random sample of cells to keep the eval dynamic
            cells = data["msg"]
            if len(cells) > num_cells:
                # Deterministic sampling based on time to vary across runs
                import random

                random.seed(time.time())
                return random.sample(cells, num_cells)
            return cells
    except Exception as e:
        print(f"Warning: Could not fetch Allen Data for eval: {e}")
        return []
    return []


# --- NEW: Task 3 - Real-World Knowledge Retrieval ---
class AllenBrainRealWorldTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.name = "allen_brain_knowledge"
        self.description = (
            "Test if the agent can correctly retrieve specific electrophysiological properties of real cells."
        )
        # Cache data at init to avoid spamming API during the loop
        print("Initializing AllenBrainRealWorldTask: Fetching live reference data...")
        self.reference_data = get_allen_reference_data(region="VISp", num_cells=5)

    def __len__(self):
        return len(self.reference_data)

    def get_example(self, index):
        if not self.reference_data:
            return {"instruction": "API Error: No reference data available. Skip this task.", "ground_truth": "N/A"}

        cell = self.reference_data[index]
        cell_id = cell["id"]
        cell_name = cell["name"]

        # Ground Truth: We want the agent to find the Input Resistance
        ephys = cell.get("ephys_features", [{}])[0]
        input_resistance = ephys.get("ri", 0)  # 'ri' is Input Resistance in MOhms

        return {
            "instruction": (
                f"Use your tools to query the Allen Brain Atlas. "
                f"Find the cell with Specimen ID {cell_id} (Name: {cell_name}). "
                f"What is its measured Input Resistance (in MOhms)? "
                f"Return ONLY the numeric value."
            ),
            "ground_truth": str(int(input_resistance)),  # Round to integer for easier comparison
        }

    def evaluate(self, prediction, ground_truth):
        # Extract numbers from prediction
        import re

        numbers = re.findall(r"\d+", prediction)
        if not numbers:
            return False

        predicted_val = float(numbers[0])
        truth_val = float(ground_truth)

        # Allow 10% margin of error (e.g. 150 vs 155 is acceptable)
        margin = truth_val * 0.1
        return abs(predicted_val - truth_val) <= margin
