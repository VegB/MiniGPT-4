"""
python inference.py --gpu-id ? \
                    --input_csv ./input_csv/visit_bench_single_image.csv \
                    --output_dir ./output_csv/visit_bench_single_image
"""

import argparse
import os
import random
import urllib.request
from urllib.parse import urlparse
from tqdm import tqdm
import json

import csv
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigpt4_eval.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--input_csv', type=str, default='./input_csv/visit_instructions_700.csv')
    parser.add_argument('--output_dir', type=str, default='./output_csv/')
    parser.add_argument('--model_name', type=str, default='MiniGPT-4')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    return args


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return csv_reader.fieldnames, data


def add_backslash_to_spaces(url):
    if ' ' in url:
        url = url.replace(' ', "%20")
    return url


def download_image(url, file_path):
    if args.verbose:
        print(url)
        print(file_path)
    try:
        urllib.request.urlretrieve(url, file_path)
        if args.verbose:
            print("Image downloaded successfully!")
    except urllib.error.URLError as e:
        print("Error occurred while downloading the image:", e)


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


if __name__ == '__main__':

    args = parse_args()
    cfg = Config(args)

    # check output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_csv = os.path.join(args.output_dir, f'{args.model_name.lower()}.csv')

    # ========================================
    #             Model Initialization
    # ========================================

    print('Initializing Chat')

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')

    # Read CSV file
    fieldname_list, input_data_list = read_csv_file(args.input_csv)

    output_data_list = []
    prediction_fieldname = f'{args.model_name} prediction'
    fieldname_list.append(prediction_fieldname)

    for row in tqdm(input_data_list, total=len(input_data_list), desc='predict'):
        if args.verbose:
            print(row)

        if 'Input.image_url' in row.keys():
            image_url_list = [row['Input.image_url']]
        elif 'image' in row.keys():
            image_url_list = [row['image']]
        else:
            image_url_list = list(eval(row['images']))

        if len(image_url_list) > 1:
            llm_prediction = '[SKIPPED]'

        else:
            # Init chat state
            chat_state = CONV_VISION.copy()
            user_message = row['instruction']
            image_url = add_backslash_to_spaces(image_url_list[0])
            img_list = []
            # print(f'Chat state init:\n{chat_state}')

            # Download image from url
            extension = image_url.split('.')[-1]
            img_path = os.path.join(os.getcwd(), f'tmp.{extension}')  # Create the local image file path
            download_image(image_url, img_path)

            # Upload Image
            chat.upload_img(img_path, chat_state, img_list)
            # print(f'len(img_list): {len(img_list)}')
            # print(f'Chat state after upload image:\n{chat_state}')

            # Ask
            chat.ask(user_message, chat_state)
            # print(f'Chat state after asking:\n{chat_state}')

            # Answer
            llm_prediction = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=args.num_beams,
                                    temperature=args.temperature,
                                    max_new_tokens=300,
                                    max_length=2000)[0]
            
            if args.verbose:
                print(f'Question:\n\t{user_message}')
                print(f'Image URL:\t{image_url}')
                print(f'Answer:\n\t{llm_prediction}')
                print('-'*30 + '\n')

            # Clean up
            os.remove(img_path)

        row[prediction_fieldname] = llm_prediction
        output_data_list.append(row)

        with open('tmp.json', 'w') as f:
            json.dump(output_data_list, f, indent=2)

    # Write to output csv file
    output_file = args.output_csv
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldname_list)
        csv_writer.writeheader()
        csv_writer.writerows(output_data_list)
