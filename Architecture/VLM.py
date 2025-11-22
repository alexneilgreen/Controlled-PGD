from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import evaluate
import numpy as np
import torch.nn.functional as F
from torch.nn import CosineEmbeddingLoss
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdaptiveCosineLoss():
    def __init__(self):
        self.loss = CosineEmbeddingLoss(margin=.5)
    
    def forward(self, input1, input2):
        d1max = max(input1.shape[1], input2.shape[1])
        input1 = F.pad(input1, (0, d1max - input1.shape[1])).to(device=device)
        input2 = F.pad(input2, (0, d1max - input2.shape[1])).to(device=device)
        target = -1 * torch.ones(input1.shape[1]).to(device=device)
        return self.loss(input1, input2, target)

    def __call__(self, input1, input2):
        return self.forward(input1, input2)

class VLM():
    def __init__(self, data_type: str, num_classes: int, labels:dict):
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        self.model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            dtype=torch.bfloat16,
            device_map="auto"
        )
        try:
            os.mkdir("data")
            print(f"Directory data created successfully.")
        except FileExistsError:
            print(f"Directory data already exists.")
        self.repsonses_file = "data/vlm_log.log"
        with open(self.repsonses_file, 'w') as f:
            f.write(f"beginning of log...\n")
        self.outpath = "Output/Models/VLM_" + data_type.upper()
        self.metric = evaluate.load("accuracy")
        self.labels = labels
        labelstr = ""
        for val in labels.values():
            labelstr += f"{val}, "
        if data_type == 'mnist':
            self.prompt = "In one word, what number is in the image? Do not use digits in the response, use words. Do not include punctuation"
        else:
            self.prompt = f"Given the folowing classes:\n{labelstr}\n, Please predict the image class. The answer should be one of the choices provided."

    def prompt_model(self, images):
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
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024, output_scores=True, return_dict_in_generate=True)
        return generated_ids.sequences[:, inputs.input_ids.shape[1]:-1]

    def get_logtis(self, images):
        generated_ids = self.prompt_model(images)
        return generated_ids.logits
    
    def print_text(self, images):
        generated_ids = self.prompt_model(images)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def __call__(self, x, mode='logits'):
        if mode == 'text':
            return self.print_text(x)
        else:
            generated_ids = self.prompt_model(x)
            with open(self.repsonses_file, 'a') as f:
                lines = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                f.writelines(lines)
            return generated_ids.to(device=device) #F.softmax(generated_ids.scores[0], dim=-1)
        
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
        return AdaptiveCosineLoss()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)    # Using Numpy Argmax b/c HuggingFace's converts output to numpy array
        return self.metric.compute(predictions=predictions, references=labels)
    
    def zero_grad(self):
        self.model.zero_grad()

    def translate_label(self, label):
        label = self.labels[label.item()]
        parsed = self.processor(text=label, return_tensors="pt")
        return parsed.input_ids.to(device=device)
        #messages = [
        #    {
        #        "role": "user",
        #        "content": [
        #            {"type": "text", "text": label}
        #        ],
        #    }
        #]
        #
        #inputs = self.processor.apply_chat_template(
        #    messages,
        #    return_tensors="pt",
        #    add_generation_prompt=True,
        #    tokenize=True,
        #).to(device)
        #with torch.no_grad():
        #    output = self.model(**inputs, output_hidden_states=True)
        #    # The last hidden state represents the text encoding/embedding
        #    return output.hidden_states[-1]