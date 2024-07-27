import os

import pandas as pd

OPENAI_IMAGENET_TEMPLATES = (
    lambda c: f'a bad depth photo of a {c}.',
    lambda c: f'a depth photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a depth photo of the hard to see {c}.',
    lambda c: f'a low resolution depth photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad depth photo of the {c}.',
    lambda c: f'a cropped depth photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a depth photo of a hard to see {c}.',
    lambda c: f'a bright depth photo of a {c}.',
    lambda c: f'a depth photo of a clean {c}.',
    lambda c: f'a depth photo of a dirty {c}.',
    lambda c: f'a dark depth photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a depth photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a depth photo of the cool {c}.',
    lambda c: f'a close-up depth photo of a {c}.',
    lambda c: f'a black and white depth photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated depth photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright depth photo of the {c}.',
    lambda c: f'a cropped depth photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a depth photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted depth photo of a {c}.',
    lambda c: f'a blurry depth photo of the {c}.',
    lambda c: f'a depth photo of the {c}.',
    lambda c: f'a good depth photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a depth photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up depth photo of the {c}.',
    lambda c: f'a depth photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution depth photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a depth photo of the clean {c}.',
    lambda c: f'a depth photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a depth photo of a nice {c}.',
    lambda c: f'a depth photo of a weird {c}.',
    lambda c: f'a blurry depth photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated depth photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted depth photo of the {c}.',
    lambda c: f'a good depth photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a depth photo of the nice {c}.',
    lambda c: f'a depth photo of the small {c}.',
    lambda c: f'a depth photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a depth photo of the large {c}.',
    lambda c: f'a black and white depth photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark depth photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a depth photo of a cool {c}.',
    lambda c: f'a depth photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
)


# a much smaller subset of above prompts
# from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
SIMPLE_IMAGENET_TEMPLATES = (
    lambda c: f'itap of a {c}.',
    lambda c: f'a bad depth photo of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a depth photo of the large {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'art of the {c}.',
    lambda c: f'a depth photo of the small {c}.',
)


IMAGENET_CLASSNAMES = (

)


CLASSNAMES = {
    'NYUV2': (
        "bathroom", "bedroom", "bookstore", "classroom", "dining room",
        "home office", "kitchen", "living room", "office", "others"
    ),
    'SUNRGBD': (
        "bathroom", "bedroom", "classroom", "computer room", "conference room", "corridor", "dining area",
        "dining room", "discussion area", "furniture store", "home office", "kitchen", "lab", "lecture theatre",
        "library", "living room", "office", "rest space", "study space"
    ),
}
