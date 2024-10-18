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
Model weights will be coming soon.

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

### GeoQA


## Train
