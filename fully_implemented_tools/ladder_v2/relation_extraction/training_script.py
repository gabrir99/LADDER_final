import os
import sys

# Add the project root directory to the Python path, needed to make the code executable also from inside different directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ladder_v2.relation_extraction.relation_extraction_trainer import RelationExtractionTrainer

cfg = {
    "checkpoint_dir": "models/bert-large-uncased",
    "dataset": "data",
    "transformer_name": "bert-large-uncased",
    "lr": 1e-5,
    "eps": 1e-8,
    "batch_size": 8,
    "epochs": 10,
    "max_seq_length": 512,
    "task_name": "relation_extraction",
    "num_labels": 11
}
if __name__ == "__main__":
    trainer = RelationExtractionTrainer(cfg)
    trainer.train_relation_extraction_model()