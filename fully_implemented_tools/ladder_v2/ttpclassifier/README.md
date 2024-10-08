### TTP Classifier Step

The code has been refactored and cleaned without changing the functionalities in order to make it easier to asses what
really are the action performed by the tool.
The following step is basically made of two tools:
* Sentence Classification: 
* Attack Pattern Identification
* Attack Pattern Mapping to a known Framework

Three different directories have been created in order to perform the three action separately, while the script in the ttpclassifier main directory will use the 3 techniques together, and generate a attack_pattern.json file containing all the attack_patterns grouped by sentence.
```
..\env\Scripts\python ttpclassifier_step.py
```
By default the result of all the extractions are created in the ../../resources/outputs/ladder/ directory.
#### Sentence Classsification
##### Training
If you want to train a model from scrach use the training script, and add to the cfg dictionary inside the file all the needed properties based on what type of model you want to train.
You can train the script using either a virtualenv or the docker container provided in the directory.
```
..\..\env\Scripts\python training_script.py
```
##### Prediction
If you wan to extract enitities from a txt file use the following command (use --help to see how to pass non-default params)
```
..\..\env\Scripts\python prediction_script.py
```

#### Attack Pattern Identification
##### Training
If you want to train a model from scrach use the training script, and add to the cfg dictionary inside the file all the needed properties based on what type of model you want to train.
You can train the script using either a virtualenv or the docker container provided in the directory.
```
..\..\env\Scripts\python training_script.py
```
##### Prediction
If you wan to extract enitities from a txt file use the following command (use --help to see how to pass non-default params)
```
..\..\env\Scripts\python prediction_script.py
```

#### Attack Pattern Mapping
For this step it has not been created a training or predicting script.