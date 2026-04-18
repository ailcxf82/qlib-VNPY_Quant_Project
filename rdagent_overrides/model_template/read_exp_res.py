import pickle
from pathlib import Path

import pandas as pd
import qlib

qlib.init()

from qlib.workflow import R

# here is the documents of the https://qlib.readthedocs.io/en/latest/component/recorder.html

# Assuming you have already listed the experiments
experiments = R.list_experiments()

# Iterate through each experiment to find the latest recorder
experiment_name = None
latest_recorder = None
for experiment in experiments:
    recorders = R.list_recorders(experiment_name=experiment)
    for recorder_id in recorders:
        if recorder_id is not None:
            experiment_name = experiment
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
            end_time = recorder.info["end_time"]
            try:
                if end_time is not None:
                    if latest_recorder is None or end_time > latest_recorder.info["end_time"]:
                        latest_recorder = recorder
                else:
                    print(f"Warning: Recorder {recorder_id} has no valid end time")
            except Exception as e:
                print(f"Error: {e}")

# Check if the latest recorder is found
if latest_recorder is None:
    print("No recorders found")
else:
    print(f"Latest recorder: {latest_recorder}")

    metrics = pd.Series(latest_recorder.list_metrics())
    output_path = Path(__file__).resolve().parent / "qlib_res.csv"
    metrics.to_csv(output_path)
    print(f"Output has been saved to {output_path}")

    # Try to load portfolio analysis; fall back to IC metrics if PortAnaRecord failed
    try:
        ret_data_frame = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        ret_data_frame.to_pickle("ret.pkl")
        print("Portfolio analysis saved to ret.pkl")
    except Exception as e:
        print(f"PortAnaRecord not available ({e}), creating fallback ret.pkl from IC metrics")
        ic_metrics = {k: v for k, v in metrics.items() if "IC" in k or "ic" in k}
        if not ic_metrics:
            ic_metrics = metrics.to_dict()
        fallback_df = pd.DataFrame([ic_metrics])
        fallback_df.to_pickle("ret.pkl")
        print("Fallback ret.pkl created with metrics:", list(ic_metrics.keys())[:5])
