import os
import csv
import evaluate
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments
from transformers.modeling_outputs import ImageClassifierOutput
from torch import stack, argmax, tensor, nn
from pathlib import Path
import torch
import numpy as np

class BasicDataCollator:
    def __call__(self, x):
        data = stack([pair[0] for pair in x])
        labels = tensor([pair[1] for pair in x])
        return {"pixel_values": data, "labels": labels}

class ViT():
    def __init__(self, data_type: str, num_classes: int):
        super(ViT, self).__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            "WinKawaks/vit-tiny-patch16-224", num_labels=num_classes, ignore_mismatched_sizes=True)
        self.outpath = "Output/Models/ViT_" + data_type.upper()
        self.metric = evaluate.load("accuracy")
        self.collator = BasicDataCollator()
        self.data_type = data_type
        self.num_classes = num_classes

    def set_pretrain_path(self, path: str):
        self.outpath = path
        
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)    # Using Numpy Argmax b/c HuggingFace's converts output to numpy array
        return self.metric.compute(predictions=predictions, references=labels)
    
    def zero_grad(self):
        self.model.zero_grad()
    
    def get_logits(self, input):
        output = self.model(input)
        if isinstance(output, ImageClassifierOutput):
            return output.logits
        else:
            return output

    def __call__(self, x):
        return self.get_logits(x)
    
    def to(self, device):
        self.model = self.model.to(device)
        return self

    def training_loop(self, train, epochs=10, lr=0.001, test_loader=None, batch_size=64):
        """
        Train the ViT model using HuggingFace Trainer.
        
        Args:
            train: Training dataloader
            epochs: Number of training epochs
            lr: Learning rate
            test_loader: Optional test dataloader for evaluation
            batch_size: Batch size for training
        """
        self.model.train()
        
        # Determine evaluation strategy
        eval_strategy = "epoch" if test_loader is not None else "no"
        
        training_args = TrainingArguments(
            output_dir=self.outpath,
            remove_unused_columns=False,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            eval_strategy=eval_strategy,
            save_strategy="epoch",
            load_best_model_at_end=True if test_loader is not None else False,
            logging_steps=100,
        )
        
        trainer = Trainer(
            model=self.model,
            data_collator=self.collator,
            args=training_args,
            train_dataset=train.dataset,    # Trainer only accepts dataset object???
            eval_dataset=test_loader.dataset if test_loader is not None else None,
            compute_metrics=self.compute_metrics,
        )
        
        print(f"\nTraining ViT for {epochs} epochs with lr={lr}...")
        trainer.train()
        self.model.eval()
        
        # Get final metrics
        final_train_acc = 0.0
        final_test_acc = 0.0
        
        # Compute training accuracy
        train_results = trainer.evaluate(eval_dataset=train.dataset)
        final_train_acc = train_results.get('eval_accuracy', 0) * 100
        print(f"\nFinal Training Accuracy: {final_train_acc:.2f}%")
        
        # Print final evaluation if test_loader provided
        if test_loader is not None:
            results = trainer.evaluate()
            final_test_acc = results.get('eval_accuracy', 0) * 100
            print(f"\nFinal Test Accuracy: {final_test_acc:.2f}%")
        
        # Note: HuggingFace Trainer doesn't have early stopping by default
        # so early_stop is always False unless we add EarlyStoppingCallback
        
        # Save metrics to CSV
        self._save_metrics_to_csv(
            model_arch="ViT-Tiny",
            dataset=self.data_type,
            epochs=epochs,
            epochs_completed=epochs,  # Trainer always completes all epochs
            lr=lr,
            batch_size=batch_size,
            train_acc=final_train_acc,
            test_acc=final_test_acc,
            early_stop=False
        )
    
    def save(self):
        self.model.save_pretrained(self.outpath)
    
    def isPretrainOnDisk(self):
        file_path = Path(self.outpath)
        # transformer lib saves in checkpoints, not singular files
        return file_path.is_dir()

    def load(self):
        self.model = AutoModelForImageClassification.from_pretrained(self.outpath)
        self.model.eval()

    def getLoss(self):
        # Pretrained AutoModelForImageClassification always use CrossEntropyLoss
        # not queryable from model impl itself, sadly
        return nn.CrossEntropyLoss()
    
    def _save_metrics_to_csv(self, model_arch, dataset, epochs, epochs_completed, lr, batch_size, train_acc, test_acc, early_stop):
        """Save training metrics to CSV file."""
        csv_path = "Output/Models/Training_Metrics.csv"
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow([
                    'Model Architecture', 'Dataset', 'Max Epochs', 'Epochs Completed', 
                    'Learning Rate', 'Batch Size', 'Training Accuracy', 'Test Accuracy', 
                    'Early Stop Triggered'
                ])
            
            # Write metrics
            writer.writerow([
                model_arch,
                dataset.upper(),
                epochs,
                epochs_completed,
                lr,
                batch_size,
                f"{train_acc:.2f}",
                f"{test_acc:.2f}",
                "Yes" if early_stop else "No"
            ])
        
        print(f"Metrics saved to {csv_path}")