# Ladder
The code has been better writtern and organized in order to use each step separately if needed.
The following directories have been created, for each step of the pipeline:

* preprocessing (No important preprocessing step is used in Ladder)
* entity_extraction (Contains code for ner and regex extraction of entities and iocs)
* ttpclassifier (Contains the code for attack pattern extraction and identification)
* relation_extraction (Contains the code for extracting relation between entities found during the entity extraction step)

Each directory contains training and predicting script need to train and use the models that belongs to each step of the pipeline. 
A readme.md file is available in each directory in order to understand how to run each script.


## Use ladder with venv
In order to use ladder execute the following command after creating the virtualenv with the required arguments:
```
env\Scripts\python ladder.py
```

## Use ladder with docker
In order to use ladder, execute the following command in the fully_implemented_tools directory
```
docker build -f fully_implemented_tools/ladder_v2/Dockerfile -t ladder_tool . 
```
To run the Docker container interactively, use:
```
docker run --gpus all -it --name ladder_v2_container ladder_tool /bin/bash
```
Once inside the container, you can execute Python scripts. For example, to run the training script, use:
```
python ladder.py
```
The first time the code will not work because you need to download some nltk components, to solve the problem execute the following two commands:
```
python -c "import nltk; nltk.download('punkt');" 
python -c "import nltk; nltk.download('averaged_perceptron_tagger');"
```

To stop the container:
```
docker stop ladder_v2_container
```
To restart the container:
```
docker restart ladder_v2_container
```
To execute the command after restarting the container:
```
docker exec -it ladder_v2_container /bin/sh
```
To remove the container:
```
docker rm ladder_v2_container
```
To view the logs of a container, use:
```
docker logs ladder_v2_container
```
If you make changes inside the container and want to save them, you can commit the changes to a new Docker image:
```
docker commit ladder_v2_container ladder_tool
```
If the files that have been generated in the container are not visible on the host machien use the following command to copy the directory containing the output of the tool:
```
docker cp ladder_v2_container:/app/fully_implemented_tools/resources/outputs/ladder  C:\dev\università\tesi\shenouda_thesis_code\fully_implemented_tools\resources\outputs
```

- first path: container path to copy on host machine
- second path: path on the host machine where to copy the files
### Debug with docker
Run the Docker container, exposing the port for debugging:

```
docker run -it --rm --name ladder_v2_container \
    -v $(pwd)/fully_implemented_tools/ladder_v2:/app/fully_implemented_tools/ladder_v2 \
    -p 5678:5678 \
    ladder_tool
```
Explanation of the options:

- -it: Runs the container in interactive mode with a terminal.
- --rm: Automatically removes the container when it exits.
- --name ladder_v2_container: Names the container for easy reference.
- -v $(pwd)/fully_implemented_tools/ladder_v2:/app/fully_implemented_tools/ladder_v2: Mounts the current directory into the container, allowing you to see changes made in the host.
- -p 5678:5678: Maps port 5678 of the host to port 5678 of the container (used for debugging).
- ladder_tool: The name of the Docker image to run.

Install debugpy: Ensure that debugpy is installed in your container. You can add it to your requirements-docker.txt or install it manually:
```
pip install debugpy
```

Modify your Python script (e.g., train.py) to listen for the debugger. Add these lines at the start of your script:
```
import debugpy

# Allow other computers to attach to the debugger
debugpy.listen(("0.0.0.0", 5678))

# Pause the program until a remote debugger is attached
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
```
Set up the launch.json file to allow VS Code to attach to the running container. This file should be located in the .vscode directory within your project:

- Go to the Debug view in VS Code by clicking on the bug icon or pressing Ctrl+Shift+D.

- Click on the gear icon to open the launch.json configuration.

- Add a new configuration to attach to the Python debugger:
```
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Remote Attach",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "/app/fully_implemented_tools/ladder_v2"
        }
      ]
    }
  ]
}
```
- connect.host: Set to localhost because you're connecting to a container running on the local machine.
- connect.port: The same port exposed by the container (5678).
- pathMappings.localRoot: The path to your project on your local machine.
- pathMappings.remoteRoot: The path inside the container where your code is located.

In your Python application inside the container using the standard Python command. The application will start and wait for a debugger to attach because of the debugpy.wait_for_client() call.

1. In VS Code, go to the Debug view (Ctrl+Shift+D).

2. Select Python: Remote Attach from the dropdown and click the green play button to start debugging.

3. Set breakpoints in your code where you want the execution to pause.

Once the debugger is attached, you will be able to step through your code, inspect variables, and use all debugging features as if you were debugging locally.

# ATTENTION
The following tool need to be run using python 3.8

