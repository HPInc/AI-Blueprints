# Super Resolution
<!-- ![alt](../../../images/BERT_QA_image.png) -->
<!-- colocar uma imagem aqui -->

In this template, our objective is to increase the resolution of images, that is, to increase the number of pixels in the image, using the FSRCNN model, a convolutional neural network model that offers greater run time speed, which receives a low-resolution image and returns a higher-resolution image that is x times larger.

 ---

 ## 1. Project Setup on AI Studio

 For this experiment, **we highly recommend yo to create a custom workspace on AI Studio using the Deep Learning GPU-based image**. 
 
 For the memory requirements, we suggest to have **at least 4GB of RAM, as well as 4GB for GPU dedicated RAM.**

 Similar configurations for a CPU-only experiment will work, but we do not recommend due to the excessive training time.

This experiment requires the DIV2K dataset to run, that should be downloaded from s3://dsp-demo-bucket/div2k-data into an assset called DIV2K.

[1. Project Setup on AI Studio](#1-project-setup-on-ai-studio)

---

 ## 2. Local deployment on AI Studio

The local deployment should be done through the Deployments tab in AIStudio. Simply select the previously trained model, and then you will be able to perform super-resolution inferences on new images.

 ---