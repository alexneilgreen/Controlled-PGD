from transformers import AutoModelForImageClassification, Trainer, TrainingArguments
from transformers.modeling_outputs import ImageClassifierOutput
from torch import stack, argmax, tensor, nn
from pathlib import Path
import evaluate

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
        self.outpath =  "pretrained/vitb10c" + data_type
        self.metric = evaluate.load("accuracy")
        self.collator=BasicDataCollator()

    def set_pretrain_path(self, path: str):
        self.outpath = path
        
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        # convert the logits to their predicted class
        predictions = argmax(logits, axis=-1)
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

    def training_loop(self, train):
        self.model.train()
        training_args = TrainingArguments(
            output_dir=self.outpath,
            remove_unused_columns=False,
            per_device_train_batch_size=64,
            eval_strategy="no",
        )
        trainer = Trainer(
            model=self.model,
            data_collator=self.collator,
            args=training_args,
            train_dataset=train.dataset, #Trainer only accepts dataset object???
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        self.model.eval()
    
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