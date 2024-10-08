

from ner_trainer import TrainerNER

cfg = {
    "checkpoint_dir": "models/xlm-roberta-large",
    "dataset": "dataset/150",
    "transformer_name": "xlm-roberta-large",
    "lr": 1e-6,
    "batch_size": 32,
    "epochs": 20,
    "max_seq_length": 128,
}
if __name__ == "__main__":
    trainer = TrainerNER(cfg)
    trainer.train_transformer_model()