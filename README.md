# <div align="center">Spaceway-AI</div>
The Spaceway-AI is a scenario-based model.  
Follow the [link](https://colab.research.google.com/drive/1Azzdfx2dv7fd8yL_djiW1EV9wrxqumiw?usp=sharing) to see the model on colab.

## 
- The model is scenario based model which takes image and then explain about the scene in the image.
- We have used the wikipedia-api for dynamic scene description.
```
pip install wikipedia-api
```
- To know the labels of the images, we are using Places365 datasets.
```
https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt
```
- We are dynamically fetching the scene description from wikipedia.
- ```
  def get_scene_description_dynamic(scene_name):
    email = ""

    user_agent = f"SceneRecognitionApp/1.0 (contact: {email})"
    wiki = wikipediaapi.Wikipedia("en", headers={"User-Agent": user_agent})
    page = wiki.page(scene_name)
    if page.exists():
        return page.summary[:500]
    else:
        return "No dynamic description available for this scene."
  ```
  while running the model, place your email ID in the contact. The length of the scene description is 500 words.
- We are using the pre-trained model for Place365 dataset which is :
```
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
```
- The model weights are available on this url
- ```
  http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar
  ```
- We are preprocessing the image in processess_image() function
- And predicting scene and provide imformation in the function predict_scen()
- And then running the script by setting construct to main.
- The input images is :

![forest](https://github.com/user-attachments/assets/da10f8c1-f4da-44e4-b5c8-d522c39651ce)
