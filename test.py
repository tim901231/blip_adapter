import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)

raw_image = Image.open("./image.jpg")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# generate caption
print(model)
caption = model.generate({"image": image})
# ['a large fountain spewing water into the air']
print(caption)

for n, p in model.named_parameters():
    if 'adapter' in n:
        print(n)
    else:
        p.requires_grad = False