from transformers import AutoProcessor, AutoModelForUniversalSegmentation
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import numpy as np

class MySequential(torch.nn.Sequential):
    def __init__(self, *args):
        super(MySequential, self).__init__(*args)
        self.register_buffer('gradients', None)

    def forward(self, x):
        # Clear the gradients buffer
        self.gradients = None

        x = self.grad_heatmap(x)
        return x
    
    def save_gradient(self, grad):
        self.gradients = grad
    
    def grad_heatmap(self, x):
        
        mask_probs, class_queries_logits = x

        num_classes = class_queries_logits.shape[-1] - 1
        
        uniform_class_queries_logits = torch.full_like(class_queries_logits, (num_classes - 1) / num_classes)

        uniform_class_weight, _ = torch.nn.functional.softmax(uniform_class_queries_logits, dim=-1).max(-1)

        class_weight, pred = torch.nn.functional.softmax(class_queries_logits, dim=-1).max(-1)

        elems, counts = torch.unique(pred, return_counts=True)
        
        # normalization
        for batch in range(class_weight.shape[0]):
            for i, c in enumerate(elems):
                max_mask = torch.where(pred[batch] == c)
                class_weight[batch][max_mask] /= counts[i]        

        weight = mask_probs * class_weight.view(mask_probs.shape[0], mask_probs.shape[1], 1, 1)

        del uniform_class_weight, uniform_class_queries_logits, num_classes, class_weight

        weight.register_hook(self.save_gradient)

        return weight
    

def compute_saliency(image):

    processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")

    model.to(device)

    features = {}

    panoptic_inputs = processor(images=[image], task_inputs=["panoptic"], return_tensors="pt")

    panoptic_inputs.to(device)

    with torch.no_grad():
        outputs = model(**panoptic_inputs)

    del model
    del processor
    del panoptic_inputs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Finished model inference. Starting postprocessing")

    class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

    mask_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    test = torch.nn.functional.interpolate(
        mask_probs[0].unsqueeze(0), size=image.size[::-1], mode="bilinear", align_corners=False
    )[0]

    mean_logits = test.mean(axis=0).cpu()

    low, high = np.percentile(mean_logits, [1, 99])

    clipped_mean_logits = np.clip(mean_logits, low, high)

    normalized_clipped_mean_logits = (clipped_mean_logits - clipped_mean_logits.min()) / (clipped_mean_logits.max() - clipped_mean_logits.min())

    normalized_clipped_mean_logits = (1 - normalized_clipped_mean_logits) * 255

    normalized_clipped_mean_logits = normalized_clipped_mean_logits.numpy().astype(np.uint8)
    
    return normalized_clipped_mean_logits, None
    plt.imshow(normalized_clipped_mean_logits)
    plt.show()
    
    batch_size = mask_probs.shape[0]
    num_queries = mask_probs.shape[1]
    num_classes = class_queries_logits.shape[-1] - 1
    height = mask_probs.shape[2]
    width = mask_probs.shape[3]
    
    model_input_mask_probs = mask_probs.clone().detach().requires_grad_(True)

    model_mini = MySequential()

    print(f"Compute gradient norms")

    # WITH GRAD ENABLED
    pred_scores_simple_head = model_mini((model_input_mask_probs, class_queries_logits))

    pred_scores, pred_labels = torch.nn.functional.softmax(class_queries_logits, dim=-1).max(-1)
    
    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(pred_scores_simple_head, mask_probs)

    model_mini.zero_grad()

    loss.backward()

    gradients = model_mini.gradients

    norm = torch.sqrt(torch.norm(gradients, dim=1, p=2)).cpu()

    del model_mini
    del model_input_mask_probs

    scaled_OoD_feature_detector = torch.nn.functional.interpolate(
        norm.unsqueeze(0), size=image.size[::-1], mode="bilinear", align_corners=False
    )[0]
    
    # this should now measure the average norm of the gradients according to the model’s predicted distribution reweighted to reflect the model output bias
    # Same meaure as https://arxiv.org/pdf/2404.10124 (REGrad) which takes the l2 norm and takes the squareroot of  https://arxiv.org/pdf/2205.10439 (EXGRAG) to mitigate overconfident results, where a high
    # model probability overshadows the contributions from gradients of other classes, potentially leading to an overemphasis. This might neglect important insights about the
    # model’s decision boundaries and uncertain regions indicated by other class gradient
    # We apply a rescaling of the logits before taking the gradinent based on the predicted labels's prominence across the features representing it to approximate a mitigation for model training bias, without calculating mean and var across the whole training set (https://arxiv.org/pdf/2107.11264)

    OoD_feature_detector_inter = torch.mean(scaled_OoD_feature_detector, dim=0).detach().numpy()

    OoD_feature_detector_inter_min, OoD_feature_detector_inter_max = OoD_feature_detector_inter.min(), OoD_feature_detector_inter.max()
    OoD_feature_detector_inter_normalized = (OoD_feature_detector_inter - OoD_feature_detector_inter_min)/(OoD_feature_detector_inter_max - OoD_feature_detector_inter_min)
    
    #plt.imsave(f'saliency_OOD_objects.png', OoD_feature_detector_inter_normalized)

    print("Computing face segmentation")

    # https://huggingface.co/mattmdjaga/segformer_b2_clothes/tree/main
    #processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    #model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    # https://huggingface.co/jonathandinu/face-parsing
    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")


    inputs = processor(images=np.array(image), return_tensors="pt")

    with torch.no_grad():

        outputs = model(**inputs)

    del model
    del inputs

    face_prediction = outputs.logits.cpu() # (batch_size, config.num_labels, logits_height, logits_width)

    # interpolate to original size
    face_prediction = torch.nn.functional.interpolate(
        face_prediction,
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    face_prediction_segmented = face_prediction.argmax(dim=1)[0]

    face_prediction = torch.median(face_prediction[0], dim=0).values.detach().numpy() # mean without .values!

    face_prediction_min, face_prediction_max = face_prediction.min(), face_prediction.max()
    face_prediction_normalized = (face_prediction - face_prediction_min)/(face_prediction_max - face_prediction_min)

    return OoD_feature_detector_inter_normalized, face_prediction_normalized



if __name__ == '__main__':

    image_name = "pexels-ruadh-cheung-20846878"

    path = "backend/input_images/" + image_name + ".jpg"

    i_m = image_name + ".jpg"

    if not os.path.exists(path):
        path = "backend/input_images/" + image_name + ".png"

        i_m = image_name + ".png"
    
    if not os.path.exists(path):
        raise Exception("Couldn't find the specified image.")
    
    path = "backend/input_images/" + i_m
    
    image = Image.open(path)
 
    processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")

    model.to(device)

    features = {}

    panoptic_inputs = processor(images=[image], task_inputs=["panoptic"], return_tensors="pt")

    panoptic_inputs.to(device)

    with torch.no_grad():
        outputs = model(**panoptic_inputs)

    del model
    del processor
    del panoptic_inputs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Finished model inference. Starting postprocessing")

    class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

    mask_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    batch_size = mask_probs.shape[0]
    num_queries = mask_probs.shape[1]
    num_classes = class_queries_logits.shape[-1] - 1
    height = mask_probs.shape[2]
    width = mask_probs.shape[3]
    
    model_input_mask_probs = mask_probs.clone().detach().requires_grad_(True)

    model_mini = MySequential()

    print(f"Compute gradient norms")

    # WITH GRAD ENABLED
    pred_scores_simple_head = model_mini((model_input_mask_probs, class_queries_logits))

    pred_scores, pred_labels = torch.nn.functional.softmax(class_queries_logits, dim=-1).max(-1)
    
    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(pred_scores_simple_head, mask_probs)

    model_mini.zero_grad()

    loss.backward()

    gradients = model_mini.gradients

    norm = torch.sqrt(torch.norm(gradients, dim=1, p=2)).cpu().detach().numpy()

    del model_mini
    del model_input_mask_probs

    scaled_OoD_feature_detector = torch.nn.functional.interpolate(
        norm.unsqueeze(0), size=image.size[::-1], mode="bilinear", align_corners=False
    )[0]
    
    # this should now measure the average norm of the gradients according to the model’s predicted distribution reweighted to reflect the model output bias
    # Same meaure as https://arxiv.org/pdf/2404.10124 (REGrad) which takes the l2 norm and takes the squareroot of  https://arxiv.org/pdf/2205.10439 (EXGRAG) to mitigate overconfident results, where a high
    # model probability overshadows the contributions from gradients of other classes, potentially leading to an overemphasis. This might neglect important insights about the
    # model’s decision boundaries and uncertain regions indicated by other class gradient
    # We apply a rescaling of the logits before taking the gradinent based on the predicted labels's prominence across the features representing it to approximate a mitigation for model training bias, without calculating mean and var across the whole training set (https://arxiv.org/pdf/2107.11264)

    OoD_feature_detector_inter = torch.mean(scaled_OoD_feature_detector, dim=0)

    OoD_feature_detector_inter_min, OoD_feature_detector_inter_max = OoD_feature_detector_inter.min(), OoD_feature_detector_inter.max()
    OoD_feature_detector_inter_normalized = (OoD_feature_detector_inter - OoD_feature_detector_inter_min)/(OoD_feature_detector_inter_max - OoD_feature_detector_inter_min)
    
    plt.imsave(f'saliency_OOD_objects.png', OoD_feature_detector_inter_normalized)

    print("Computing face segmentation")

    # https://huggingface.co/mattmdjaga/segformer_b2_clothes/tree/main
    #processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    #model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    # https://huggingface.co/jonathandinu/face-parsing
    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")


    inputs = processor(images=np.array(image), return_tensors="pt")

    with torch.no_grad():

        outputs = model(**inputs)

    del model
    del inputs

    face_prediction = outputs.logits.cpu().detach().numpy() # (batch_size, config.num_labels, logits_height, logits_width)

    # interpolate to original size
    face_prediction = torch.nn.functional.interpolate(
        face_prediction,
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    face_prediction_segmented = face_prediction.argmax(dim=1)[0]

    face_prediction = torch.median(face_prediction[0], dim=0).values # mean without .values!

    face_prediction_min, face_prediction_max = face_prediction.min(), face_prediction.max()
    face_prediction_normalized = (face_prediction - face_prediction_min)/(face_prediction_max - face_prediction_min)

    plt.imsave(f'saliency_face_parsing_predicton.png', face_prediction_normalized)
