import json
import os

# Example usage
command_string = "python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_DisCo_massfromgap_smooth_shed1 general.subfolder=gauss_corr_4_gap_twinturbo_usem_addgapmass/ do_train_template=0"
launch_json_path = "/home/users/o/oleksiyu/WORK/hyperproject/.vscode/launch.json"  # Update this path accordingly

def update_launch_json(command_string, launch_json_path):
    # Split the command string to extract the script and arguments
    parts = command_string.split()
    program = parts[0]
    args = parts[1:]

    # Prepare the configuration object to insert/replace
    new_config = {
        "name": "Python Debugger: Command Translator",
        "type": "python",
        "request": "launch",
        "program": program,
        "console": "integratedTerminal",
        "args": args
    }

    # Read the existing launch.json file
    with open(launch_json_path, 'r') as file:
        data = json.load(file)

    # Check if "configurations" exists and is a list
    if "configurations" not in data or not isinstance(data["configurations"], list):
        print("Error: launch.json file is not in the expected format.")
        return

    # Replace the existing "Python Debugger: Command Translator" or add if not present
    for i, config in enumerate(data["configurations"]):
        if config.get("name") == "Python Debugger: Command Translator":
            data["configurations"][i] = new_config
            break
    else:
        # If the "Python Debugger: Command Translator" configuration doesn't exist, add it
        data["configurations"].append(new_config)

    # Write the updated content back to the launch.json file
    with open(launch_json_path, 'w') as file:
        json.dump(data, file, indent=4)

    print("launch.json has been updated successfully.")

update_launch_json(command_string, launch_json_path)