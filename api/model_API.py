from fastapi import FastAPI, Request
import torch
import uvicorn
import re
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Optional

from predict_model import SVGGenerator

# Create instance
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize model
MODEL_PATH = "/Users/jianglimeng/Desktop/projects/fine_tune/tuned_codellama"
TOKENIZER_PATH = "/Users/jianglimeng/Desktop/projects/fine_tune/tokenizer_svg_extended"

svg_generator: Optional[SVGGenerator] = None

@app.on_event("startup")
async def startup_event():
    # load the model only once
    global svg_generator
    print("Loading model...")
    try:
        svg_generator = SVGGenerator(MODEL_PATH, TOKENIZER_PATH)
        print("Model loaded.")
    except Exception as e:
        print(f"Failed: {e}")

# response treated as HTML
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Returns an HTML page build on request.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    """
    Get user input, call the model to make predictions, and return the results.
    """
    if svg_generator is None:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Model not loaded."})
    
    # get user input
    form_data = await request.form()
    user_text = form_data.get("user_text", "")

    if not user_text:
        return templates.TemplateResponse("index.html", {"request": request, "svg_output": "", "user_text": "Image description needed."})

    # call the model
    # make 2 predictions
    try:
        svg_output1 = svg_generator.predict(user_text)
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error in prediction: {e}"})
    try:
        svg_output2 = svg_generator.predict(user_text)
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error in prediction: {e}"})

    # display the model prediction
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "svg_output_1": svg_output1, "svg_output_2": svg_output2, "user_text": user_text}
    )

# Create the HTML files
if __name__ == "__main__":
    # format: file_name: instance_name
    uvicorn.run("model_API:app", host="0.0.0.0", port=8000, reload=True)