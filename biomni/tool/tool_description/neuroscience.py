description = [
    {
        "name": "analyze_spike_train_statistics",
        "description": "Analyzes a neuronal spike train to calculate firing rate, inter-spike intervals (ISI), and coefficient of variation (CV). Generates an ISI histogram.",
        "required_parameters": [
            {
                "default": None,
                "name": "spike_times",
                "type": "list or numpy.ndarray",
                "description": "List of time stamps (in seconds) when spikes occurred.",
            }
        ],
        "optional_parameters": [
            {
                "default": None,
                "name": "duration",
                "type": "float",
                "description": "Total duration of the recording in seconds.",
            },
            {"name": "output_dir", "type": "str", "description": "Directory to save result plots.", "default": "./"},
        ],
    },
    {
        "name": "get_brain_region_metadata",
        "description": "Retrieves metadata, IDs, and hierarchy information for a specific brain region.",
        "required_parameters": [
            {
                "default": None,
                "name": "region_name",
                "type": "str",
                "description": "Common name or acronym of the brain region (e.g., 'Hippocampus', 'VISp').",
            }
        ],
    },
]
