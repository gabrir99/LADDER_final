## Project Organization
The code has been organized in two main directories:

* external_code
* fully_implemented_tools

The external_code directories contains all the code, found on github or other platforms that fully or partially implements the tool described in one of the papers of interest.

The fully_implemented_tools directory contains all the code of the implemented tool starting from the available found code or from scratch.

Each tool is saved in its own directory and contains a readme.md with the informations needed for the execution of the tool.

The output of each tool it is saved under the "resources/name_of_the_tool" directory under the fully_implemented_tools directory.

## Usefull command for cuda and gpu uses
Use this command to see the installed cuda version.
```
nvcc --version
```
Use this command to see the GPU usage per process.
```
nvidia-smi
```

## Analyze the output of training, testing and validation using tensorboard
To analyze the log file use the following command if you are using the virtual envirnonment:
```
py -3.8 -m tensorboard.main --logdir logs\experiment_1
```

## Formatting the code
Use the following command to install the black package:
```
pip install black
```
In order to format files or directories use the following commands:
```
black <filename>
black <directory>
```
## Trouble Shooting Cuda Errors
* https://saturncloud.io/blog/why-torchcudaisavailable-returns-false-even-after-installing-pytorch-with-cuda/


## Setup Environment
You can use either the docker container and execute the code inside it, or you can create a virtualenv and install the requirements.txt file and execute the code using the instruction available inside each readme.md in each directory.

### Python Virtual Environments Tips
In order to create an environment and make the use of the code easier use the following command.
This will create a new virtual environment named myenv in the current directory.
```
python3 -m venv myenv
```
On Windows, venv creates a batch file called activate.bat located in the following directory, execute the following code in the cmd.
```
myenv\Scripts\activate
```
Once the virtualenv is up use the following command to install the needed packages
```
pip install <package-name>
```
Once you installed all the needed packages use the following command to freeze all the used packages in a requiremnts.txt file.
```
pip freeze > requirements.txt
```
In order to deactivate the virtual enviroment execute the following code in the cmd.
```
myenv\Scripts\deactivate
```

Use the following command to see the available python versions
```
py --list
```
Use this if you want to execute a script using a different python verssion
```
py -'version_number'
```

#### Create the venv with the needed requiremnts

When someone else clones the repository, they can set up the virtual environment and install the dependencies using:
```
python -m venv myenv
\myenv\Scripts\activate
\myenv\Scripts\python -m pip install -r requirements.txt
```

### Docker Tips
#### Build docker image
After creating your Dockerfile and requirements.txt, build the Docker image using the following command in the terminal:
```
docker build -f path/to_docker_file -t your-image-name .
```
This command will create a Docker image with Python, CUDA support, and all the necessary libraries.

#### Run the Docker Container
Now you can run your Docker container using the following command:
```
docker run --gpus all -v /your/local/directory:/app your-image-name
```
- --gpus all: Enables GPU support within the container.
- -v /your/local/directory:/app: Mounts your local project directory to the container's /app directory.

If you want to override the default command (CMD) in the Dockerfile and specify your Python script to run, you can add it to the end of the docker run command:
```
docker run --gpus all -v /your/local/directory:/app your-image-name python train.py
```
If the following command is not working just start bin/sh in the container and move to the directory containining the script code.
```
docker run -it <image_name> /bin/sh
```

#### Verify CUDA Support Inside the Container
To check if CUDA is available in your container, you can run the following command:
```
docker run --gpus all -it your-image-name  python -c "import torch; print(torch.cuda.is_available())"  
```
This should output information about your GPU devices.

#### Debug Script Inside Container

To connect to an already running Docker container and debug a Python application using Visual Studio Code (VS Code), follow these steps:

##### Find the Running Container
First, you need to identify the container you want to connect to.

Open a terminal and list the running Docker containers:

```
docker ps
```

This will display a list of running containers. Identify the container's name or ID that you want to connect to.

##### Ensure the Container is Running the Debugger
To debug the Python application in the container, ensure that the container is running a Python process with debugpy (or a similar remote debugging tool). If it's not already set up, you can attach to the container and start the debugger manually.

Option 1: If Debugger is Already Running
If the container was started with debugpy (or another Python debugger) listening on a specific port (e.g., 5678), you can proceed to connect from VS Code.

Option 2: Manually Start the Debugger Inside the Container
If the debugger is not yet running, you can attach to the container and start it:

Attach to the container:

```
docker exec -it <container_name_or_id> /bin/bash
Replace <container_name_or_id> with the container's name or ID.

Navigate to the directory where your Python script is located and start the debugger:

```
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client your_script.py
This command starts the Python application and waits for the VS Code debugger to attach on port 5678.

##### Configure VS Code to Attach to the Container
In VS Code, you need to set up a remote debugging configuration.

Open your project in VS Code.

Go to the Debug panel (Ctrl + Shift + D).

Click on the gear icon to configure the launch settings, and choose "Python: Remote Attach."

Modify the launch.json configuration file as follows:

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
          "remoteRoot": "/app"
        }
      ]
    }
  ]
}
```

- host: Set to localhost because you're connecting to the container locally.
- port: Set to the port where debugpy is listening (e.g., 5678).
- localRoot: Path to your code on your local machine.
- remoteRoot: Path to your code inside the Docker container.

##### Attach the Debugger
With the container running and the debugger listening, go back to VS Code.
In the Debug panel, select the "Python: Remote Attach" configuration.
Click the green "Start Debugging" button or press F5.
VS Code will connect to the Python process running inside the container, and you can start debugging as if it were running locally.

##### Debug Your Application
Once connected, you can set breakpoints, inspect variables, and step through your code just as you would in a local debugging session.

