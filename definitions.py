import os

ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
WANDB_KEY = '9f4a033fffce45cce1ee2d5f657d43634a1d2889'
ATTENTION_CMAP = 'inferno'
model_save_dir = os.path.join(ROOT_FOLDER, 'models')
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)