import torch
from torchvision import transforms
from PIL import Image

def model_fn(model_dir):
    device = torch.device("cpu")  # Use CPU for inference
    model = torch.jit.load(f"{model_dir}/model.pth", map_location=device)
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    """
    Deserialize and preprocess input data.
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return torch.tensor(data)
    elif request_content_type == "image/jpeg":
        # Handle image input
        image = Image.open(io.BytesIO(request_body))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    with torch.no_grad():
        predictions = model(input_data)
    return predictions

def output_fn(prediction, content_type):
    """
    Serialize the output into JSON format.
    """
    if content_type == "application/json":
        return json.dumps(prediction.tolist())
    else:
        raise ValueError(f"Unsupported response content type: {content_type}")
