import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class SVGGenerator:
    """
    Convert a natural language text description to an Scalable Vector Graphics (SVG) string.
    """
    def __init__(self, model_path: str, tokenizer_path: str):
        """
        Initialize model: load base model, LoRA adapters and merge them.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

        # resize the model embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.eval()

    def predict(self, description: str, max_new_tokens: int = 6400) -> str:
        """
        Predict the SVG string based on the input text description
        """
        # following the training prompt pattern, create the prompt
        prompt = f"Given the following description: {description}, generate the corresponding SVG string.\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # SVG string generation
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True, # generate new contents for each .generate call
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        # process the generated text
        # remove the prompt part
        svg_string = generated_text.replace(prompt, "").strip()

        # cleaning
        start_tag = '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 384 384">'
        end_tag = '</svg>'

        # extract the real SVG content
        match = re.search(r'<svg.*?<\/svg>', svg_string, re.DOTALL)
        if match:
            final_svg_string = match.group(0)
        else:
            # reformat the generated text
            print("Cleaning the generated content...")
            final_svg_string = svg_string
            if not final_svg_string.startswith('<svg'):
                final_svg_string = start_tag + final_svg_string
            if not final_svg_string.endswith('</svg>'):
                final_svg_string += end_tag

        return final_svg_string