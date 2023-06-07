import gradio as gr
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, Blip2ForConditionalGeneration, VisionEncoderDecoderModel
import torch
import open_clip

from huggingface_hub import hf_hub_download

# Microsoft GIT (short for GenerativeImage2Text) model, large-sized version, fine-tuned on COCO
git_processor_large_coco = AutoProcessor.from_pretrained("microsoft/git-large-coco")
git_model_large_coco = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")
# Microsoft GIT (short for GenerativeImage2Text) model, large-sized version, fine-tuned on TextCaps
git_processor_large_textcaps = AutoProcessor.from_pretrained("microsoft/git-large-r-textcaps")
git_model_large_textcaps = AutoModelForCausalLM.from_pretrained("microsoft/git-large-r-textcaps")


# Salesforce Model card for image captioning pretrained on COCO dataset - base architecture (with ViT large backbone).
blip_processor_large = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model_large = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


coca_model, _, coca_transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

git_model_large_coco.to(device)
git_model_large_textcaps.to(device)
blip_model_large.to(device)

coca_model.to(device)

def generate_caption(processor, model, image, tokenizer=None, use_float_16=False):
    inputs = processor(images=image, return_tensors="pt").to(device)

    if use_float_16:
        inputs = inputs.to(torch.float16)
    
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)

    if tokenizer is not None:
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
   
    return generated_caption


def generate_caption_coca(model, transform, image):
    im = transform(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(im, seq_len=20)
    return open_clip.decode(generated[0].detach()).split("<end_of_text>")[0].replace("<start_of_text>", "")


def generate_multiple_captions(image):

    caption_git_large_coco = generate_caption(git_processor_large_coco, git_model_large_coco, image)

    caption_git_large_textcaps = generate_caption(git_processor_large_textcaps, git_model_large_textcaps, image)


    caption_blip_large = generate_caption(blip_processor_large, blip_model_large, image)


    caption_coca = generate_caption_coca(coca_model, coca_transform, image)


    return  caption_git_large_coco, caption_git_large_textcaps, caption_blip_large, caption_coca

def generate_single_caption(image):
    caption_git_large_coco = generate_caption(git_processor_large_coco, git_model_large_coco, image)

    return caption_git_large_coco