from flask import Flask, render_template, request
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, PreTrainedTokenizerFast
import torch
import io
import base64

app = Flask(__name__)

# Load the model, feature extractor, and tokenizer once during startup
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = PreTrainedTokenizerFast.from_pretrained("distilgpt2")

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            try:
                # Open and process the image
                img = Image.open(file)
                pixel_values = vit_feature_extractor(images=img, return_tensors="pt").pixel_values
                encoder_outputs = model.generate(pixel_values.to('cpu'), num_beams=5)
                generated_sentences = tokenizer.batch_decode(encoder_outputs, skip_special_tokens=True)
                generated_caption = generated_sentences[0].split('.')[0]

                # Convert image to Base64 for displaying in HTML
                base64_image = image_to_base64(img)

                return render_template('index.html', caption=generated_caption, base64_image=base64_image)

            except Exception as e:
                return render_template('index.html', error=f'Error processing the image: {str(e)}')

    return render_template('index.html', caption=None, base64_image=None, error=None)

if __name__ == '__main__':
    app.run(debug=True)
