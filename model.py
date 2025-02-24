import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image


class Model:
    def __init__(self, settings_path: str = './settings.yaml'):
        # Load settings from YAML file
        with open(settings_path, "r") as file:
            self.settings = yaml.safe_load(file)

        # Initialize model settings
        self.device = self.settings['model-settings']['device']
        self.model_name = self.settings['model-settings']['model-name']
        self.threshold = self.settings['model-settings']['prediction-threshold']
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)

        # Initialize label settings
        self.labels = self.settings['label-settings']['labels']
        self.labels_ = ['a photo of ' + label for label in self.labels]  # Improve model accuracy
        self.default_label = self.settings['label-settings']['default-label']

        # Precompute text features for faster prediction
        self.text_features = self.vectorize_text(self.labels_)

    @torch.no_grad()
    def transform_image(self, image: np.ndarray):
        """Convert numpy image to preprocessed tensor."""
        pil_image = Image.fromarray(image).convert('RGB')
        tf_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        return tf_image

    @torch.no_grad()
    def tokenize(self, text: list):
        """Tokenize text for CLIP model."""
        text = clip.tokenize(text).to(self.device)
        return text

    @torch.no_grad()
    def vectorize_text(self, text: list):
        """Encode text into feature vectors."""
        tokens = self.tokenize(text=text)
        text_features = self.model.encode_text(tokens)
        return text_features

    @torch.no_grad()
    def predict_(self, text_features: torch.Tensor, image_features: torch.Tensor):
        """Compute similarity between image and text features."""
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        values, indices = similarity[0].topk(1)
        return values, indices

    @torch.no_grad()
    def predict(self, image: np.array) -> dict:
        """
        Predict the label for an input image.

        Args:
            image (np.array): Input image as a numpy array with RGB channel ordering.

        Returns:
            dict: Contains the predicted label and confidence score.
                  Example: {'label': 'some_label', 'confidence': 0.X}
        """
        tf_image = self.transform_image(image)
        image_features = self.model.encode_image(tf_image)
        values, indices = self.predict_(text_features=self.text_features, image_features=image_features)

        label_index = indices[0].cpu().item()
        model_confidence = abs(values[0].cpu().item())
        label_text = self.default_label if model_confidence < self.threshold else self.labels[label_index]

        return {
            'label': label_text,
            'confidence': model_confidence
        }

    @staticmethod
    def plot_image(image: np.array, title_text: str):
        """Plot an image with a title."""
        plt.figure(figsize=[13, 13])
        plt.title(title_text)
        plt.axis('off')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)