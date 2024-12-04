import os
import torch
from torchvision import models, transforms
from PIL import Image
import requests
import wikipediaapi
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_places365_labels():
    url = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
    response = requests.get(url)
    if response.status_code == 200:
        class_labels = []
        for line in response.text.strip().split("\n"):
            label = line.split(" ")[0].split("/")[-1].replace("_", " ")
            class_labels.append(label)
        return class_labels
    else:
        raise Exception("Failed to download Places365 labels.")

class_labels = download_places365_labels()
print("Scene labels downloaded")

def load_places365_model(device):
    model = models.resnet50(pretrained=False)
    num_classes = 365
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    weights_url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
    weights_path = "resnet50_places365.pth.tar"
    if not os.path.exists(weights_path):
        print("Downloading pre-trained weights...")
        response = requests.get(weights_url)
        with open(weights_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    model = model.to(device)
    model.eval()
    return model

model = load_places365_model(device)
print("Weights downloaded")

def get_scene_description_dynamic(scene_name):
    email = ""

    user_agent = f"SceneRecognitionApp/1.0 (contact: {email})"
    wiki = wikipediaapi.Wikipedia("en", headers={"User-Agent": user_agent})
    page = wiki.page(scene_name)
    if page.exists():
        return page.summary[:500]
    else:
        return "No dynamic description available for this scene."

def download_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        raise Exception(f"Failed to download image from URL: {url}")

def preprocess_image(image_source, is_url=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if is_url:
        image = download_image_from_url(image_source)
    else:
        image = Image.open(image_source).convert("RGB")
    return transform(image).unsqueeze(0)

def predict_scene(image_source, model, class_labels, device, is_url=False):
    image_tensor = preprocess_image(image_source, is_url=is_url).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        scene_name = class_labels[predicted_idx.item()]
        description = get_scene_description_dynamic(scene_name)
        return scene_name, description

if __name__=='__main__':
  image_source = "https://images.unsplash.com/photo-1448375240586-882707db888b?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
  is_url = image_source.startswith("http://") or image_source.startswith("https://")

  if not is_url and not os.path.exists(image_source):
      raise Exception(f"Image file {image_source} not found!")

  scene_name, description = predict_scene(image_source, model, class_labels, device, is_url=is_url)
  print(f"Scene: {scene_name}")
  print(f"Description: {description}")


