import subprocess
import yaml
import time
import os
import tempfile
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="examples/logs/DroneVLM2MultiMap/gradnav_moe/NLP/gate_mid/06-06-2025-16-11-12/final_policy.pt",
        help="Path to checkpoint passed to train_gradnav_vlm_2_moe.py"
    )
    args = parser.parse_args()

    # Path setup
    yaml_path = "examples/cfg/gradnav_vlm_2_moe/drone_multi_map.yaml"
    script_path = "examples/train_gradnav_vla_moe.py"
    checkpoint_path = args.checkpoint

    # Load the original YAML ONCE
    with open(yaml_path, "r") as f:
        original_data = yaml.safe_load(f)

    map_list = ["gate_mid", "gate_left"]

    for map_id in map_list:
        # Copy the data in memory
        data = yaml.safe_load(yaml.dump(original_data))
        data["params"]["config"]["map_name"] = map_id

        # Loop through task 0 to 3
        for task_id in tqdm(range(4), desc=f"Running tasks on {map_id}", ncols=100):
            # Modify the task field
            data["params"]["config"]["player"]["task"] = task_id

            # Write to a temporary YAML file
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmpfile:
                yaml.dump(data, tmpfile, sort_keys=False)
                tmp_yaml_path = tmpfile.name

            # Build command
            cmd = [
                "python",
                script_path,
                "--cfg", tmp_yaml_path,
                "--checkpoint", checkpoint_path,
                "--play",
                "--render",
            ]

            # Launch the process and wait for it to finish
            process = subprocess.Popen(cmd)
            process.wait()

            # Optional: small delay between runs
            time.sleep(1)

            # Clean up the temporary YAML file
            os.remove(tmp_yaml_path)

    print("All tasks finished, original config untouched!")


if __name__ == "__main__":
    main()
