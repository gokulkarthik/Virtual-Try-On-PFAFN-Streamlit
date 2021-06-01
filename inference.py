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
from tqdm.auto import tqdm

opt = TestOptions().parse()

# list human-cloth pairs
with open('demo.txt', 'w') as file:
    lines = [f'input.png {cloth_img_fn}\n' for cloth_img_fn in os.listdir('dataset/test_clothes')]
    file.writelines(lines)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('[INFO] Data Loaded')

warp_model = AFWM(opt, 3)
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, opt.warp_checkpoint)
print('[INFO] Warp Model Loaded')

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
gen_model.eval()
gen_model.cuda()
load_checkpoint(gen_model, opt.gen_checkpoint)
print('[INFO] Gen Model Loaded')

def get_result_images():

    result_images = []
    for i, data in tqdm(enumerate(dataset)):

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

        a = real_image.float().cuda()
        b= clothes.cuda()
        c = p_tryon

        combine = torch.cat([b[0], c[0]], 2).squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        rgb = (cv_img * 255).astype(np.uint8)

        result_images.append(rgb)


    return result_images