---
title:  'Monitoring'
sidebar_position: 4
---

# Using the Monitoring Page 

AI Studio runs the tools you select natively, so you can use popular data science solutions in a single application. AI Studio supports the open-source tools ML Flow (for pytorch) and Tensorboard (for TensorFlow). 

## To begin monitoring:

1. From your project’s home page, select and run the workspace you want to use. 

2. Select a Jupyter Notebook from the root folder. 

3. Import the necessary MLFlow or Tensorboard packages into the notebook.

4. Embed MLFlow or Tensorboard hooks for tool to capture, track, and manage metrics and artifacts.

    :::tip
    
    See the quick guide use case for an example of how to work with MLFlow and Tensorboard in AI Studio. Visit [MLflow](https://mlflow.org/docs/latest/index.html) and [Tensorboard](https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks) documentation sites for detailed information on these tools. 

    :::

5. Run the notebook you selected. 

5. Navigate to the *Monitor* tab and use the integrated application to monitor your data as usual.
    
    :::tip
    
    Click the icon in the upper-right corner of the DS tool panel to enter full screen view.
    
    :::

You have real-time access to your project’s CPU, GPU, VRAM, and memory consumption by the Z by HP logo so you can check on your running models at any time. AI Studio periodically recommends configuration improvements based on your tool usage trends.
