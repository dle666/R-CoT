<p align="center">
    <img src="https://s21.ax1x.com/2024/10/17/pAUm9qH.png" width="80" style="margin-bottom: 0.2;"/>
<p>

<h3 align="center"> <a href="">R-CoT: Reverse Chain-of-Thought Problem Generation for Geometric Reasoning in Large Multimodal Models</a></h3>
<h2></h2>

<h5 align="center"> Please give us a star ⭐ for the latest update.  </h5>

<h5 align="center">


## News 
Data and code will be coming soon.


## Dataset
You can download the training and testing data used by R-CoT from [R-CoT_Data](https://huggingface.co/datasets/dle666/R-CoT).

## 🐳 Model Zoo
| Model|Vision Part|Language Model|Transformers(HF)|
|------------|------------------------|--------------------|-------------------------------------------------------|
|R-CoT-8B    |InternViT‑300M‑448px    |internlm2_5‑7b‑chat |[🤗R-CoT-8B](https://huggingface.co/dle666/R-CoT-8B)    |
|R-CoT-7B    |EVA-CLIP                |InternLM-Chat-7B    |[🤗R-CoT-7B](https://huggingface.co/dle666/R-CoT-7B)    |
|R-CoT-2B    |InternViT‑300M‑448px    |internlm2-chat-1_8b |[🤗R-CoT-2B](https://huggingface.co/dle666/R-CoT-2B)    |

## Environment
### NPU
```python
pip install --upgrade deepspeed
pip install torchvision==0.16.0
pip install torch==2.1.0
pip install transformers==4.32.0
pip install torch_npu==2.1.0
```
### GPU
```python
conda create -n rcot python=3.9 -y
conda activate rcot
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation
```


## Evaluation
### Mathista (geometry problem solving)
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
python model_vqa.py --checkpoint weight_path
```

Run automatic evaluation to calculate the accuracy:
```python
python geo_acc_calculate.py --predictions_file path-to-output-file
```

## Train
