---
title:  'Model Deployment Overview'
sidebar_position: 1
---
# Model Deployment  

Model Services are embedded utilities that let users deploy registered models for local inference. Locally published model services are hosted on localhost as a browser tab. Each tab represents a different notebook and you can simultaneously run as many models as your machine can handle.

## To deploy a new model service:

1. From the project's *Overview* page, select the ***Deployments*** tab, then click **Deploy Your Model**.

2. Name your new service and select one of the registered models from your connected MLflow account.
   
    :::note
    Only models saved to ML Flow can be deployed.  See [Registering a New Model - Scikit-learn](/docs/aistudio/using-aistudio/model-deployment/register-model-scikit-learn.md) for more information.
    :::

3. Choose your model version and GPU configuration.

4. Select a workspace to deploy the service on, then click **Deploy**.

    :::note
    You can’t add models to a running workspace, so add any models you might find useful ***before*** you run it.
    :::

When you create a new service, it automatically becomes viewable from the *Deployments* tab. Use the more options icon at the end of each row to edit or delete a deployment at any time. 

 :::tip
 
 Your project’s GPU, CPU, VRAM, and memory consumption appear in the corner of the screen, so you can visualize the effects of the tests you run in real-time.
 
 :::