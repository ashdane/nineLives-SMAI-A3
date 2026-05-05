# Tomato Leaf Disease Diagnosis App 🍅

Hey there! This is my project for predicting tomato leaf diseases from pictures. The main idea was to build something farmers could actually use, so there's a Streamlit frontend attached to our machine learning backend.

## How the Models Work
I ended up going with a two-stage approach because in real life, users might accidentally upload random images (like selfies or a picture of a tractor). 

1. **Zero-Shot Filtering**: First, the image goes through a pre-trained CLIP model (`clip-vit-base-patch32`). This acts as a gatekeeper. It relies on zero-shot classification to check if the image is actually a tomato leaf or something totally unrelated.
2. **Disease Classifier (Fine-Tuning)**: If it passes the first check, the image moves on to an `efficientnet_b0` model. I didn't train this from scratch—it's a pretrained model that I fine-tuned. For the first epoch, the backbone is frozen to just warm up the new classification head. From epoch 2 onwards, I unfreeze the entire network and drop the learning rate so we don't wreck the pretrained weights. This fine-tuning was done on a 10-class subset of the PlantVillage dataset.

## Evaluation & Analysis
When testing the EfficientNet model on our validation split, the progressive fine-tuning approach worked surprisingly well. Freezing the backbone for the very first epoch helped stabilize the early loss significantly. Once the backbone unfroze, the accuracy jumped up fast and smoothed out. I did notice that the model can sometimes struggle a little bit when distinguishing between Early Blight and Late Blight since the spots look visually similar in their early stages. But the confidence scores usually reflect that uncertainty, which is why the UI shows the top 5 predictions.

The CLIP model's zero-shot performance is honestly great out of the box. It almost never lets a picture of a human or a notebook pass through. It can sometimes get tricked by leaves of other plants if they look too much like a tomato leaf, which makes sense since we only gave it fairly basic text prompts to work with. 

Overall, the pipeline feels pretty robust. Combining a large, generalized model (CLIP) to handle the noisy inputs with a small, specialized one (EfficientNet) for the actual task gave the best of both worlds without needing a massive GPU to serve the app.

## How to Run It
Make sure you have your environment set up. You can check `requirements.txt` for the packages you need.

```bash
# To grab the dataset (it'll automatically extract to data/tomato)
python scripts/download_data.py

# To train the model yourself
python train.py --data_dir data/tomato --epochs 5 --batch_size 32 --lr 1e-3

# Run the frontend web app
streamlit run app.py
```

## Folder Setup
I tried to keep the directory fairly clean:
- `app.py`: The Streamlit application.
- `train.py`: The training loop and validation code.
- `disease_info.json`: Text data for the UI so it can give actionable treatment advice.
- `scripts/`: Helper scripts for downloading the data and making dummy models.
- `notebooks/`: My rough colab experiments.
- `checkpoints/`: Where the trained `.pth` weights end up.

## Acknowledgments
I definitely got some help putting this codebase together. Big thanks to large language models like Google Gemini, OpenAI ChatGPT, and Anthropic Claude. They were super useful for debugging weird PyTorch tensor shape errors, helping me structure the Streamlit layout properly, and brainstorming the two-stage CLIP approach.
