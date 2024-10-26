<p align="center">
    <img src="https://s21.ax1x.com/2024/10/17/pAUm9qH.png" width="80" style="margin-bottom: 0.2;"/>
<p>

<h3 align="center"> <a href="https://arxiv.org/abs/2410.17885">R-CoT: Reverse Chain-of-Thought Problem Generation for Geometric Reasoning in Large Multimodal Models</a></h3>
<h2></h2>

<h5 align="center"> Please give us a star ⭐ for the latest update.  </h5>

https://github.com/user-attachments/assets/b271b472-d478-4f52-b743-d62526c53781

## News 
* 🎉🎉🎉 We source the GeoMM dataset.
* 🎉🎉🎉 We source the model weights for R-CoT-8B, R-CoT-7B, R-CoT-2B.
* 🎉🎉🎉 We source the evaluation code.
* 🎉🎉🎉 We source the training code.
* 🎉🎉🎉 We release the paper [R-CoT](https://arxiv.org/abs/2410.17885)


## Dataset
You can download the training and testing data used by R-CoT from [R-CoT_Data](https://huggingface.co/datasets/dle666/R-CoT).

Examples of GeoMM:
<br>
<p align="center">
    <img src="https://s21.ax1x.com/2024/10/20/pAaGpRJ.png" width="800"/>
<p>
<br>

    
## 🐳 Model Zoo

<div align="center">

|   Model Name   |    Vision Part      |     Language Model      |       Transformers (HF)    |  MathVista(Geo)  |  GeoQA  |
|:-----------:|:-------------------------:|:------------------------------------:|:------------------------------------:|:-----------:|:-----------:|
|  **R-CoT-8B**  | InternViT‑300M‑448px   | internlm2_5‑7b‑chat                  | [🤗R-CoT-8B](https://huggingface.co/dle666/R-CoT-8B) |  75.0  |  75.1  |
|  **R-CoT-7B**  | EVA-CLIP               | InternLM-Chat-7B                     | [🤗R-CoT-7B](https://huggingface.co/dle666/R-CoT-7B) |  62.5  |  68.2  |
|  **R-CoT-2B**  | InternViT‑300M‑448px   | internlm2-chat-1_8b                  | [🤗R-CoT-2B](https://huggingface.co/dle666/R-CoT-2B) |  57.7  |  62.6  |
| **R-CoT-Qwen** | Vit-BigG               |  Qwen-7B                             | [🤗R-CoT-Qwen](https://huggingface.co/dle666/R-CoT-Qwen) | 50.5 | 57.0 |

</div>


## Environment
### GPU
```python
conda create -n rcot python=3.9 -y
conda activate rcot
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation
```

### NPU
```python
pip install --upgrade deepspeed
pip install torchvision==0.16.0
pip install torch==2.1.0
pip install transformers==4.32.0
pip install torch_npu==2.1.0
```

### Modify code to adapt to NPU
Needs to be added in a training script (e.g. finetune.py):
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```
Replace --bp16 with --fp16 in sh scripts and weight config files.

## Evaluation
### MathVista (geometry problem solving)
You need to download the test image [MathVista_test.zip](https://huggingface.co/datasets/dle666/R-CoT). Unzip and rename it to "images" and place it in the path MathVista_eval/data.

We give the response generation scripts for the different models, they start with "generate_response_geo", here R-CoT-7B is used as an example:
```python
cd MathVista_eval/evaluation
python generate_response_geo_rcot7b.py -output_dir ../results --output_file output_bard.json --checkpoint weight_path
```

Extract the short answer text for score calculation:
```python
python extract_answer.py --output_dir ../results --output_file output_bard.json 
```

Calculate the final score:
```python
python calculate_score.py --output_dir ../results --output_file output_bard.json --score_file scores.json
```

### GeoQA
You need to download the test image [GeoQA_test.zip](https://huggingface.co/datasets/dle666/R-CoT). Unzip and rename it to "test" and place it in the path GeoQA_test/images/test.
Generate responses from the model:
```python
cd GeoQA_test
python model_vqa.py --checkpoint weight_path
```

Run automatic evaluation to calculate the accuracy:
```python
python geo_acc_calculate.py --predictions_file path-to-output-file
```

## Train
The json file used for R-CoT training can be downloaded at [Link](https://huggingface.co/datasets/dle666/R-CoT). Please change the image path in the json file to your path and put the image under your path.

For R-CoT-8B:
You need to place the downloaded 'rcot8b_rcot2b_training_json' under the path set in 'shell/data/rcot_finetune.json'
```python
cd R-CoT8B-main
sh shell/R-CoT-8B/rcot8b_finetune_full.sh
```

For R-CoT-7B:
You need to place the downloaded 'GeoMM.json' and 'geo170k.json' under the path set in 'data.txt'
```python
cd R-CoT7B-main
sh finetune.sh
```

For R-CoT-2B:
You need to place the downloaded 'rcot8b_rcot2b_training_json' under the path set in 'shell/data/rcot_finetune.json'
```python
cd R-CoT2B-main
sh shell/R-CoT-2B/rcot2b_finetune_full.sh
```

## Acknowledgement
R-CoT focuses on generating high-quality mathematical inference data to improve the inference performance of models. R-CoT is based on QwenVL, InternVL2, and InternLM-XC2. Thanks to [Qwen-VL](https://github.com/QwenLM/Qwen-VL.git), [InternVL](https://github.com/OpenGVLab/InternVL), [InternLM-XC2](https://github.com/InternLM/InternLM-XComposer) and [LLaVA](https://github.com/haotian-liu/LLaVA).

## Copyright
R-CoT project is intended for non-commercial use only. For commercial inquiries or to explore more advanced versions of the R-CoT series LMMs, please contact Prof. Yuliang Liu at ylliu@hust.edu.cn.
