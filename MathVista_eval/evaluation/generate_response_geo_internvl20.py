import os
import io
import time
import argparse
import sys
import types

from tqdm import tqdm
import torch
import sys
sys.path.append('../')
from utilities import *

from build_query import create_query_data
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM 

# import torch_npu
# from torch_npu.contrib import transfer_to_npu
# torch_npu.npu.set_device("npu:0")

def verify_response(response):
    if isinstance(response, str):
        response = response.strip() 
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout
    
    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e
    
    # Restore the original stdout
    sys.stdout = old_stdout
    
    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()
    
    # Return the captured output or error
    return captured_output, error
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='../results/1018_internvl20_checkpoint2-2800')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    # model
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='llm engine',
                        choices = ['gpt-3.5-turbo', 'claude-2', 'gpt4', 'gpt-4-0613', 'bard'])
    parser.add_argument('--key', type=str, default='', help='key for llm api')
    # query
    parser.add_argument('--query_file', type=str, default=None)  
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_bard.json') 
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')   
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type', 
                        choices = ['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')
    # other settings
    parser.add_argument('--rerun', action='store_true', help='rerun answer extraction for all problems')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    # load data
    input_file = os.path.join(args.data_dir, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)
    
    # load or create query data
    if args.query_file:
        query_file = os.path.join(args.data_dir, args.query_file)
        if os.path.exists(query_file):
            print(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
    else:
        print("\nCreating new query...")
        # load caption
        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                print(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    print("Caption data loaded.")
                except:
                    print("Caption data not found!! Please Check.")                    
        # load ocr
        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                print(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    print("OCR data loaded.")
                except:
                    print("OCR data not found!! Please Check.")
        # create query
        query_data = create_query_data(data, caption_data, ocr_data, args)

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)
    
    # load results
    if os.path.exists(output_file):
        print("\nResults already exist.")
        print(f"Reading {output_file}...")
        results = read_json(output_file)
    else:
        results = {}

    path = args.checkpoint
    kwargs = {'device_map': 'auto'} 
    model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval().cuda()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    print(use_thumbnail)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(
                do_sample=False,
                top_k=50,
                top_p=0.9,
                num_beams=5,
                max_new_tokens=1024,
                eos_token_id=tokenizer.eos_token_id,
            )
    print(f"Model loaded.")
    
    # build final test pid list
    test_pids = list(data.keys())
    print("\nNumber of test problems in total:", len(test_pids))

    skip_pids = []
    if not args.rerun:
        print("\nRemoving problems with existing valid response...")
        for pid in test_pids:
            # print(f"Checking {pid}...")
            if pid in results and 'response' in results[pid]:
                response = results[pid]['response']
                if verify_response(response):
                    # print(f"Valid response found for {pid}.")
                    skip_pids.append(pid)
    else:
        print("\nRerun answer extraction for all problems...")

    test_pids = [pid for pid in test_pids if pid not in skip_pids]
    print("Number of test problems to run:", len(test_pids))
    # print(test_pids)

    # tqdm, enumerate results
    for _, pid in enumerate(tqdm(test_pids)):
        problem = data[pid]
        query = query_data[pid]
        if problem['metadata']['task'] == 'geometry problem solving':
            query = "<image>\nFirst perform reasoning, then finally select the question from the choices in the following format: Answer: xxx." + query.replace("Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nSolution:", "")
            print(query)
            image = problem['image']
            image_path = os.path.join(args.data_dir, image)
            pixel_values = load_image(image_path, image_size).to(torch.bfloat16).cuda()
            # import pdb
            # pdb.set_trace()
            if args.debug:
                print("--------------------------------------------------------------")
            print(f"\nGenerating response for {pid}...")
            # response = model.get_response(image_path, query)
            response = model.chat(tokenizer=tokenizer, pixel_values=pixel_values, question=query, generation_config=generation_config, verbose=True)
            # print(f"Response: {response}")
            
            results[pid] = problem
            results[pid]['query'] = query
            if args.shot_type == 'solution':
                results[pid]['response'] = response
            else:
                output, error = evaluate_code(response)
                results[pid]['response'] = response
                results[pid]['execution'] = output
                results[pid]['error'] = str(error)
            if args.debug:
                print(f"\n#Query: \n{query}")
                print(f"\n#Response: \n{response}")
        
            try:
                print(f"Saving results to {output_file}...")
                save_json(results, output_file)
                print(f"Results saved.")
            except Exception as e:
                print(e)
                print(f"Error in saving {output_file}")
