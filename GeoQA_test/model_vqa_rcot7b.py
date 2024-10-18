import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

# from gllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from gllava.conversation import conv_templates, SeparatorStyle
# from gllava.model.builder import load_pretrained_model
# from gllava.utils import disable_torch_init
# from gllava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM 

from PIL import Image
import math

import torch_npu
from torch_npu.contrib import transfer_to_npu

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    checkpoint_path = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True).eval().cuda()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")][0:args.test_size]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = '/root/paddlejob/workspace/env_run/dlluo/dle/G-LLaVA-main/images/' + line["image"]
        qs = line["text"]
        qs = "<ImageHere>" + qs
        cur_prompt = qs

        response, _ = model.chat(tokenizer, query=qs, image=image_file, history=[], do_sample=False)

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
    parser.add_argument("--answers-file", type=str, default="results/answer1.jsonl")
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
