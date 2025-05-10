# Description

In this folder, we move forward with valuable examples Deep Learning applications on AI Studio, focusing mainly in Computer Vision with Convolutional Neural Networks and Natural Language with Transformers. Currently, we are working with three examples, all of them requiring Deep Learning workspaces (preferably with GPU) to run.

## 1. Bert QA
This experiment shows a simple BertQA experiment, providing code to train a model, and other to load a trained model from Hugging Face, deploying a service in MLFlow to perform the inference

## 2. Text Generation
This experiment shows how to create a simple text generation, one character per time. This example uses a dataset of Shakespeare's texts.

## 3. Super Resolution
This is a Computer Vision experiment that uses convolutional networks for image transformation - more specifically improving the resolution of an image. This experiment requires the DIV2K dataset to run, that should be downloaded from s3://dsp-demo-bucket/div2k-data into an assset called DIV2K.