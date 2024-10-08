import os
import sys

# Add the project root directory to the Python path, needed to make the code executable also from inside different directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from ladder_v2.ttpclassifier.sentence_classification.sentence_classification_trainer import SentenceClassificationTrainer

cfg = {
    "checkpoint_dir": "models/bert-base-uncased",
    "dataset": "data",
    "transformer_name": "bert-base-uncased",
    "lr": 1e-5,
    "dropout": 0.5,
    "decay": 0.0,
    "epochs": 20,
    "max_seq_length": 64,
    "task_name": "atk-pattern",
    "fine_tune": True, #whenever to fine-tune embedding or not
}
if __name__ == "__main__":
    trainer = SentenceClassificationTrainer(cfg)
    trainer.train_classification_model()