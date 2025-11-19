import subprocess
import yaml
import time
import os
import tempfile
from tqdm import tqdm
import argparse


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to checkpoint passed to train_gradnav_vla_moe.py"
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint

    # Path setup
    yaml_path = "examples/cfg/gradnav_vla_moe/drone_long_task.yaml"
    script_path = "examples/train_gradnav_vla_moe.py"

    with open(yaml_path, "r") as f:
        original_data = yaml.safe_load(f)

    for task_id in tqdm(range(8), desc="Running tasks", ncols=100):
        data = yaml.safe_load(yaml.dump(original_data))
        data["params"]["config"]["player"]["task"] = task_id
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmpfile:
            yaml.dump(data, tmpfile, sort_keys=False)
            tmp_yaml_path = tmpfile.name

        cmd = [
            "python",
            script_path,
            "--cfg", tmp_yaml_path,
        ]

        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])

        cmd.extend(["--play", "--render"])
        process = subprocess.Popen(cmd)
        process.wait()
        time.sleep(1)

        # Clean up the temporary YAML file
        os.remove(tmp_yaml_path)

    print("All tasks finished, original config untouched!")


if __name__ == "__main__":
    main()
