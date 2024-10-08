import os
import sys

# Add the project root directory to the Python path, needed to make the code executable also from inside different directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from ladder_v2.ttpclassifier.attack_pattern_identification.entity_extraction_trainer import AtkPatternEntityExtractionTrainer

cfg = {
    "checkpoint_dir": "models/roberta-base",
    "dataset": "data",
    "transformer_name": "roberta-base",
    "lr": 1e-5,
    "dropout": 0.5,
    "decay": 0.0,
    "batch_size": 16,
    "epochs": 30,
    "max_seq_length": 256,
    "task_name": "entity_recognition",
    "freeze_bert": False,
    "gradient_clip": -1,
    "lstm_dim": -1, 
    "fine_tune": True, #whenever to fine-tune embedding or not
}
if __name__ == "__main__":
    trainer = AtkPatternEntityExtractionTrainer(cfg)
    trainer.train_entity_extraction_model()