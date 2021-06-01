import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F

import io
from PIL import Image
from flask import Flask, jsonify, request
from tqdm.auto import tqdm

app = Flask(__name__)

opt = TestOptions().parse()

# list human-cloth pairs
with open('demo.txt', 'w') as file:
    lines = [f'input.png {cloth_img_fn}\n' for cloth_img_fn in os.listdir('dataset/test_clothes')]
    file.writelines(lines)

warp_model = AFWM("", 3)
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, 'checkpoints/PFAFN/warp_model_final.pth')

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
gen_model.eval()
gen_model.cuda()
load_checkpoint(gen_model, 'checkpoints/PFAFN/gen_model_final.pth')


def save_cloth_transfers(image_bytes):

    opt_name = 'demo'
    opt_batchSize = 1

    image = Image.open(io.BytesIO(image_bytes))
    image.save('dataset/test_img/input.png')

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    start_epoch, epoch_iter = 1, 0

    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    step = 0
    step_per_batch = dataset_size / opt_batchSize

    for epoch in range(1, 2):
        for i, data in tqdm(enumerate(dataset, start=epoch_iter)):
            iter_start_time = time.time()
            total_steps += opt_batchSize
            epoch_iter += opt_batchSize

            real_image = data['image']
            clothes = data['clothes']
            ##edge is extracted from the clothes image with the built-in function in python
            edge = data['edge']
            edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
            clothes = clothes * edge

            flow_out = warp_model(real_image.cuda(), clothes.cuda())
            warped_cloth, last_flow, = flow_out
            warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                                        mode='bilinear', padding_mode='zeros')

            gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
            gen_outputs = gen_model(gen_inputs)
            p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            m_composite = m_composite * warped_edge
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

            path = 'results/' + opt_name
            os.makedirs(path, exist_ok=True)
            sub_path = path + '/PFAFN'
            os.makedirs(sub_path, exist_ok=True)

            if step % 1 == 0:
                a = real_image.float().cuda()
                b = clothes.cuda()
                c = p_tryon
                combine = torch.cat([a[0], b[0], c[0]], 2).squeeze()
                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(sub_path + '/' + str(step) + '.jpg', bgr)

            step += 1
            if epoch_iter >= dataset_size:
                break

    return True


@app.route('/predict')
def predict():
    if request.method == 'POST':
        print('#'*100)
        file = request.files['file']
        image_bytes = file.read()
        save_cloth_transfers(image_bytes=image_bytes)
        return jsonify({'status': True})
    else:
        return jsonify({'message': "Only accept POST requests"})


if __name__ == '__main__':
    app.run()


