### Relation Extraction Step

The code has been refactored and cleaned without changing the functionalities in order to make it easier to asses what really are the action performed by the tool.

##### Training
If you want to train a model from scrach use the training script, and add to the cfg dictionary inside the file all the needed properties based on what type of model you want to train.
You can train the script using either a virtualenv or the docker container provided in the directory.
```
..\env\Scripts\python training_script.py
```
##### Prediction
If you wan to extract relations from a json file that already contains the extracted entities for each sentence use the following command
```
..\env\Scripts\python relation_extraction_step.py
```