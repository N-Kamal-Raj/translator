from flask import Flask, request, render_template
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

app = Flask(__name__)

# Load model and tokenizer once
model_name = "facebook/mbart-large-50-one-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX")

@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    if request.method == "POST":
        input_text = request.form["text"]
        inputs = tokenizer(input_text, return_tensors="pt")
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["ta_IN"]
        )
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return render_template("index.html", translation=translation)

if __name__ == "__main__":
    app.run(debug=True)
