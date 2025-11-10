from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
FILE IS HEAVILY WIP, DEFINITELY DOES NOT WORK AS IS
'''

class VLM():
    def __init__(self, data_type: str, num_classes: int, labels:str):
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
        self.model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.outpath = "Output/Models/VLM_" + data_type.upper()
        self.metric = evaluate.load("accuracy")
        self.prompt = f"<image> For this image and the following labels:\n{labels}, Please predict the image class. The answer should be in one word and in cases where there is a collision, choose the most likely class."\
                        " Additionally, in cases where there is no classification for this image, please output \"Uknown\""

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
        generated_ids.logits
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(generated_texts[0])

    def __call__(self, x):
