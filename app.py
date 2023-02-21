import json
import numpy as np
import torch
from transformers import pipeline


class InferlessPythonModel:
    def initialize(self):
        self.generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

    def infer(self, prompt):
        pipeline_output = self.generator(prompt, do_sample=True, min_length=50)
        generated_txt = pipeline_output[0]["generated_text"]
        return generated_txt

    def finalize(self):
        self.pipe = None
