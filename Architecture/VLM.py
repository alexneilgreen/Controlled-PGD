from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import evaluate
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
FILE IS HEAVILY WIP, DEFINITELY DOES NOT WORK AS IS
'''

class VLM():
    def __init__(self, data_type: str, num_classes: int, labels:dict):
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        self.model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.outpath = "Output/Models/VLM_" + data_type.upper()
        self.metric = evaluate.load("accuracy")
        self.prompt = f"<image> For this image and the following labels:\n{labels}, Please predict the image class. The answer should be in one word and in cases where there is a collision, choose the most likely class."\
                        " Additionally, in cases where there is no classification for this image, please output \"Unknown\""

    def prompt(self, images):
        conversation = []
        for image in images:
            conversation.append(
            {
                "role": "user",
                "content":[
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt}
                ]
            }
            )
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            ).to(device, dtype=torch.bfloat16)
        generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=100)
        return generated_ids

    def get_logtis(self, images):
        generated_ids = self.prompt(images)
        return generated_ids.logits
    
    def print_text(self, images):
        generated_ids = self.prompt(images)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def __call__(self, x, mode='logits'):
        if mode == 'text':
            return self.print_text(x)
        else:
            return self.get_logtis(x)
        
    def training_loop(self, train, epochs=10, lr=0.001, test_loader=None, batch_size=64):
        print('Classfication Fine-Tuning disabled for VLM!')

    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def save(self):
        pass
    
    def isPretrainOnDisk(self):
        return True
    
    def load(self):
        pass

    def getLoss(self):
        # Pretrained AutoModelForImageClassification always use CrossEntropyLoss
        # not queryable from model impl itself, sadly
        return torch.nn.CrossEntropyLoss()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)    # Using Numpy Argmax b/c HuggingFace's converts output to numpy array
        return self.metric.compute(predictions=predictions, references=labels)
    
    def zero_grad(self):
        self.model.zero_grad()