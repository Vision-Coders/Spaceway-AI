import os
import torch
from torchvision import models, transforms
from PIL import Image
import requests
import wikipediaapi

# Function to download class labels and scene descriptions from Places365 dataset
def download_places365_labels():
    url = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
    response = requests.get(url)
    if response.status_code == 200:
        class_labels = []
        for line in response.text.strip().split("\n"):
            label = line.split(" ")[0].split("/")[-1].replace("_", " ")  # Extract scene label
            class_labels.append(label)
        return class_labels
    else:
        raise Exception("Failed to download Places365 labels.")

# Dynamically fetch scene descriptions using Wikipedia
def get_scene_description_dynamic(scene_name):
    user_agent = "SceneRecognitionApp/1.0 (contact: kshitijbudholiya2006@gmail.com)"  # Replace with your email or website
    wiki = wikipediaapi.Wikipedia("en", headers={"User-Agent": user_agent})
    page = wiki.page(scene_name)
    if page.exists():
        # Return the summary of the Wikipedia page
        return page.summary[:500]  # Limit to 500 characters
    else:
        return "No dynamic description available for this scene."


# Load pre-trained model for Places365
def load_places365_model(device):
    model = models.resnet50(pretrained=False)
    num_classes = 365
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Download the Places365 weights
    weights_url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
    weights_path = "resnet50_places365.pth.tar"
    if not os.path.exists(weights_path):
        print("Downloading pre-trained weights...")
        response = requests.get(weights_url)
        with open(weights_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")

    # Load the weights
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    model = model.to(device)
    model.eval()
    return model

# Preprocess input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Predict scene and provide information
def predict_scene(image_path, model, class_labels, device):
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        scene_name = class_labels[predicted_idx.item()]
        description = get_scene_description_dynamic(scene_name)
        return scene_name, description

# Main function
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load scene labels and model
    print("Loading scene labels...")
    class_labels = download_places365_labels()
    print("Loading pre-trained model...")
    model = load_places365_model(device)

    # Input image
    image_path = ""  # Replace with your image path
    if not os.path.exists(image_path):
        raise Exception(f"Image file {image_path} not found!")

    # Predict and display results
    print("Predicting scene...")
    scene_name, description = predict_scene(image_path, model, class_labels, device)
    print(f"Scene: {scene_name}")
    print(f"Description: {description}")
