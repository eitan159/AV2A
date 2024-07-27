# import os
#
# import pandas as pd
#
# OPENAI_IMAGENET_TEMPLATES = (
#     lambda c: f'a bad thermal infrared photo of a {c}.',
#     lambda c: f'a thermal infrared photo of many {c}.',
#     lambda c: f'a sculpture of a {c}.',
#     lambda c: f'a thermal infrared photo of the hard to see {c}.',
#     lambda c: f'a low resolution thermal infrared photo of the {c}.',
#     lambda c: f'a rendering of a {c}.',
#     lambda c: f'graffiti of a {c}.',
#     lambda c: f'a bad thermal infrared photo of the {c}.',
#     lambda c: f'a cropped thermal infrared photo of the {c}.',
#     lambda c: f'a tattoo of a {c}.',
#     lambda c: f'the embroidered {c}.',
#     lambda c: f'a thermal infrared photo of a hard to see {c}.',
#     lambda c: f'a bright thermal infrared photo of a {c}.',
#     lambda c: f'a thermal infrared photo of a clean {c}.',
#     lambda c: f'a thermal infrared photo of a dirty {c}.',
#     lambda c: f'a dark thermal infrared photo of the {c}.',
#     lambda c: f'a drawing of a {c}.',
#     lambda c: f'a thermal infrared photo of my {c}.',
#     lambda c: f'the plastic {c}.',
#     lambda c: f'a thermal infrared photo of the cool {c}.',
#     lambda c: f'a close-up thermal infrared photo of a {c}.',
#     lambda c: f'a black and white thermal infrared photo of the {c}.',
#     lambda c: f'a painting of the {c}.',
#     lambda c: f'a painting of a {c}.',
#     lambda c: f'a pixelated thermal infrared photo of the {c}.',
#     lambda c: f'a sculpture of the {c}.',
#     lambda c: f'a bright thermal infrared photo of the {c}.',
#     lambda c: f'a cropped thermal infrared photo of a {c}.',
#     lambda c: f'a plastic {c}.',
#     lambda c: f'a thermal infrared photo of the dirty {c}.',
#     lambda c: f'a jpeg corrupted thermal infrared photo of a {c}.',
#     lambda c: f'a blurry thermal infrared photo of the {c}.',
#     lambda c: f'a thermal infrared photo of the {c}.',
#     lambda c: f'a good thermal infrared photo of the {c}.',
#     lambda c: f'a rendering of the {c}.',
#     lambda c: f'a {c} in a video game.',
#     lambda c: f'a thermal infrared photo of one {c}.',
#     lambda c: f'a doodle of a {c}.',
#     lambda c: f'a close-up thermal infrared photo of the {c}.',
#     lambda c: f'a thermal infrared photo of a {c}.',
#     lambda c: f'the origami {c}.',
#     lambda c: f'the {c} in a video game.',
#     lambda c: f'a sketch of a {c}.',
#     lambda c: f'a doodle of the {c}.',
#     lambda c: f'a origami {c}.',
#     lambda c: f'a low resolution thermal infrared photo of a {c}.',
#     lambda c: f'the toy {c}.',
#     lambda c: f'a rendition of the {c}.',
#     lambda c: f'a thermal infrared photo of the clean {c}.',
#     lambda c: f'a thermal infrared photo of a large {c}.',
#     lambda c: f'a rendition of a {c}.',
#     lambda c: f'a thermal infrared photo of a nice {c}.',
#     lambda c: f'a thermal infrared photo of a weird {c}.',
#     lambda c: f'a blurry thermal infrared photo of a {c}.',
#     lambda c: f'a cartoon {c}.',
#     lambda c: f'art of a {c}.',
#     lambda c: f'a sketch of the {c}.',
#     lambda c: f'a embroidered {c}.',
#     lambda c: f'a pixelated thermal infrared photo of a {c}.',
#     lambda c: f'itap of the {c}.',
#     lambda c: f'a jpeg corrupted thermal infrared photo of the {c}.',
#     lambda c: f'a good thermal infrared photo of a {c}.',
#     lambda c: f'a plushie {c}.',
#     lambda c: f'a thermal infrared photo of the nice {c}.',
#     lambda c: f'a thermal infrared photo of the small {c}.',
#     lambda c: f'a thermal infrared photo of the weird {c}.',
#     lambda c: f'the cartoon {c}.',
#     lambda c: f'art of the {c}.',
#     lambda c: f'a drawing of the {c}.',
#     lambda c: f'a thermal infrared photo of the large {c}.',
#     lambda c: f'a black and white thermal infrared photo of a {c}.',
#     lambda c: f'the plushie {c}.',
#     lambda c: f'a dark thermal infrared photo of a {c}.',
#     lambda c: f'itap of a {c}.',
#     lambda c: f'graffiti of the {c}.',
#     lambda c: f'a toy {c}.',
#     lambda c: f'itap of my {c}.',
#     lambda c: f'a thermal infrared photo of a cool {c}.',
#     lambda c: f'a thermal infrared photo of a small {c}.',
#     lambda c: f'a tattoo of the {c}.',
# )
#
# # a much smaller subset of above prompts
# # from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
# SIMPLE_IMAGENET_TEMPLATES = (
#     lambda c: f'itap of a {c}.',
#     lambda c: f'a bad thermal infrared photo of the {c}.',
#     lambda c: f'a origami {c}.',
#     lambda c: f'a thermal infrared photo of the large {c}.',
#     lambda c: f'a {c} in a video game.',
#     lambda c: f'art of the {c}.',
#     lambda c: f'a thermal infrared photo of the small {c}.',
# )
#
# CLASSNAMES = {
#     'LLVIP': (
#         "background", "people"
#     ),
#     'FLIRV1': (
#         "bicycle", "car", "dog", "person"
#     ),
#     'FLIRV2': (
#         "bike", "bus", "car or pick-up trucks or vans", "hydrant", "traffic light", "motor", "construction equipment or trailers",
#         "person", "sign", "skateboard", "stroller or pram", "semi truck or freight truck"
#     ),
#     'LSOTB': (
#         "airplane", "badger", "bat", "bird", "boat", "bus", "car", "cat", "cow", "coyote", "deer", "dog",
#         "drone", "fox", "helicopter", "hog", "leopard", "motobike", "person", "truck"
#     )
# }


import os

import pandas as pd

OPENAI_IMAGENET_TEMPLATES = (
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
)

# a much smaller subset of above prompts
# from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
SIMPLE_IMAGENET_TEMPLATES = (
    lambda c: f'itap of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'art of the {c}.',
    lambda c: f'a photo of the small {c}.',
)

CLASSNAMES = {
    'LLVIP': (
        "background", "people"
    ),
    'FLIRV1': (
        "bicycle", "car", "dog", "person"
    ),
    'FLIRV2': (
        "bike", "bus", "car or pick-up trucks or vans", "hydrant", "traffic light", "motor", "construction equipment or trailers",
        "person", "sign", "skateboard", "stroller or pram", "semi truck or freight truck"
    ),
    'LSOTB': (
        "airplane", "badger", "bat", "bird", "boat", "bus", "car", "cat", "cow", "coyote", "deer", "dog",
        "drone", "fox", "helicopter", "hog", "leopard", "motobike", "person", "truck"
    )
}
