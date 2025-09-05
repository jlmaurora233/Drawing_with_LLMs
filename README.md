# Drawing_with_LLMs
Kaggle competition topic. The goal is to generate Scalable Vector Graphics (SVG) code that renders a text prompt as an image which can be scaled in size without quality loss. The project works towards enabling LLMs to reason about abstract descriptions and translate them into precise, executable code.

### Dependency Installation <br>
First run: <br>
`!pip install -U "transformers==4.45.2" "trl==0.9.4" "peft==0.12.0" "accelerate==0.34.2" "datasets>=2.20.0" safetensors einops lxml defusedxml cairosvg pillow scikit-image` <br>
Then run: <br>
`!pip install -q -U bitsandbytes` <br>
Now the necessary packages are installed. <br>

### Methodology)<br> 
*Data generation*: Tried different LLMs (Llama 3, Mistral, etc.) to generate text descriptions that are less than 15 words. The prompts are crafted so that they utilize the sample training data and include clear instructions on the themes they can cover. <br>
*Image generation and conversion*： Use diffusion models to generate .png images for each of the text descriptions. Then use the [processing code](https://www.kaggle.com/code/richolson/stable-diffusion-svg-scoring-metric) to simplify the image. After that, extract the utf-8 encoding data of the image and match them to the text descriptions to form pairs. <br>
*Fine-tuning*：Our goal is to fine-tune an LLM to generate SVG from brief natural language descriptions. Due to the need to generate long SVG strings (5.5k-6k tokens), we initially considered the Llama 3 series. However, due to memory constraints, we chose Code Llama, which is better suited for code-related tasks and more manageable in terms of memory, Tiny Llama, and the Llama3-8B models. Note that the versions we chose were the Instruct ones to make the model understand and respond to this specific generation task. For the fine-tuning process, we first added new tokens to the tokenizer to better handle the SVG format. Given our hardware limitations, we adopted 4-bit quantization to load the model and further reduce memory usage. My job was to fine-tune the Code Llama model. The training phase involved extensive tuning of training arguments, particularly the learning rate and optimizer, to stabilize the model's performance. Due to computing power limit, I could only train for a single epoch, so I implemented early stopping to prevent overfitting and save on computational resources. The fine-tuned model's validation loss decreased from 1.15 to 0.8. <br>
*Deployment*: Wrapped up the model as a class in `predict_model.py`. Then wrote a simple web API using FastAPI in `model_API.py`. To launch it, run `uvicorn model_API:app --reload`

### Evaluation <br>
The metric we choose is perplexity, which essentially measures how many candidate tokens the model is choosing between. We use this because it evaluates the model's core ability to understand the task (mapping descriptions to SVG syntax). The lower the perplexity score is, the better it is at predicting the next token in a sequence. The model perplexity is 3.3636 on the test dataset, indicating that the model is quite confident in choosing the next token with the SVG's "grammar". While this indicated successful training, this model's perplexity of 3.x was higher than the perplexity of 2.x achieved by my teammates' models (Tiny Llama and Llama 3). <br>
Despite the higher perplexity, the fine-tuned Code Llama model proved to be more effective for this specific task. The Tiny Llama model, due to its size, could only generate fragmented code, failing to produce complete SVG strings. <br>

**Limitation:** Model outputs contain extraneous text and hence require a post-processing step to extract the SVG code. It takes several attempts before the model can generate a clean, valid SVG string. <br>

Example: <br>
text description: a cat on a beach <br>
generated SVG: <br>
<img src="https://raw.githubusercontent.com/jlmaurora233/Drawing_with_LLMs/refs/heads/main/api/output.svg" alt="cat on a beach" width="200"/>

