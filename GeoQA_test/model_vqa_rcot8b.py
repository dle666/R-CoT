import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM 

from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess

from PIL import Image
import math

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    checkpoint_path = args.checkpoint
    kwargs = {'device_map': 'auto'} 
    model = AutoModel.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval().cuda()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    print(use_thumbnail)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(
                do_sample=False,
                top_k=50,
                top_p=0.9,
                num_beams=5,
                max_new_tokens=1024,
                eos_token_id=tokenizer.eos_token_id,
            )
    print(f"Model loaded.")

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")][0:args.test_size]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = '/root/paddlejob/workspace/env_run/dlluo/dle/G-LLaVA-main/images/' + line["image"]
        qs = line["text"]
        qs = "<image>\n" + qs
        cur_prompt = qs
        pixel_values = load_image(image_file, image_size).to(torch.bfloat16).cuda()
        response = model.chat(tokenizer=tokenizer, pixel_values=pixel_values, question=qs, generation_config=generation_config, verbose=True)
           
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": response,
                                   "answer_id": ans_id,
                                   "model_id": 'intern',
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="images/")
    parser.add_argument("--question-file", type=str, default="playground/data/test_questions.jsonl")
    parser.add_argument("--answers-file", type=str, default="results/1018_internvl20_checkpoint2-3000/answer1.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    eval_model(args)
