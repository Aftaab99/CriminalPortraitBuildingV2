import base64
from flask import Flask

from flask import request
from generate_dataset import generate_dataset
from params import dataset_path
import json
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import pickle
from torchvision.utils import save_image, make_grid

main = Flask(__name__)

with open('models/encoder_obj_latest.pkl', 'rb') as f:
    text_enc = pickle.load(f)


@main.route("/get_prediction", methods=['GET'])
def get_prediction():
    if request.method == "GET":
        # gets the string describing the features from the frontend
        # features = request.json['face_attributes']
        features = request.get_json(force=True)['face_attributes']
        # feature_str = generate_captions(set(features))
        # gets the numpy array from the model
        np_arr = get_closest_image(features)
        # Creates PIL image using numpy array
        img = Image.fromarray(np_arr)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # creates a json object with base64 string
        data = {}
        data['generated_image'] = str(to_base64(img))
        json_data = json.dumps(data)
        return json_data


def to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


# def get_prediction_from_model(s: str):
#     # return np.random.rand(3,2)
#     # TODO: Will add this function after the model is ready
#     netG = Generator()
#     netG.load_state_dict(torch.load('models/model_epoch_199_latest.pt', map_location=torch.device('cpu')))
#     netG.eval()
#     enc = InfersentEncoder()
#     inp = torch.FloatTensor(enc.encode([s])).cpu()
#
#     enc = text_enc.encode([enc])
#     encoding = torch.FloatTensor(enc)
#     fake = netG(encoding)
#
#     img = make_grid(fake, normalize=True)
#     img1 = np.transpose(img.detach().numpy(), (1, 2, 0))
#     return img1.astype(np.uint8)


def get_closest_image(attrs: list):
    import os
    s1 = set(attrs)

    d = generate_dataset()
    cur_max_len = -1
    cur_best_image = None
    loop_run_c = 0
    for k, v in d.items():
        k1 = os.path.join(dataset_path, 'img_align_celeba/img_align_celeba/{}'.format(k))

        s2 = set(v)
        s3 = s1 & s2
        if len(s3) > cur_max_len:
            cur_max_len = len(s3)
            cur_best_image = k1
        if len(s3) == len(s1):
            i = Image.open(os.path.join(dataset_path, cur_best_image))
            i.save('test.png')
            print(loop_run_c)
            return Image.open(k1)
        loop_run_c += 1
    print(cur_best_image)
    print(loop_run_c)
    i = Image.open(os.path.join(dataset_path, cur_best_image))
    i.save('test.png')
    return i


