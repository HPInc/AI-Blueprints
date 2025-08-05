---
title:  'Using a Predefined Workspace'
sidebar_position: 1
---
# Using a Predefined Workspace 

Workspaces created with AI Studio templates sync with collaborators in shared projects, so teams know they’re using workspaces in the same environments. 

### From the *Project Overview* page:

1. Select **New Workspace**.

2. Choose the workspace size that is most compatible with your machine(s). 

    - **[Minimal](/docs/aistudio/using-aistudio/workspaces/base-images#minimal-image) (CPU Ready)**: For basic libraries, small workspaces offer lightning-fast startup times, but sacrifice compute and power to do so. 

    - **[Data Science](/docs/aistudio/using-aistudio/workspaces/base-images#data-science-image) (CPU Ready)**: The recommended setup for most users. Medium workspaces that provide the most balance of compute power and speed.

    - **[Deep Learning](/docs/aistudio/using-aistudio/workspaces/base-images#deep-learning-image) (GPU Ready)**: For the most complex experiments, this is the most powerful workspace AI Studio can offer.

    - **[NVIDIA's NGC Catalog](/docs/aistudio/using-aistudio/workspaces/base-images#ngc-catalog-containers) (GPU Ready)**: Leverage hundreds of models and containers to transform AI Studio into your hub for optimized GPU Software solutions.

        :::tip
         If you choose not to download the required workspace images, you’ll need to do so before you can run your workspace for the first time.
        :::
    :::note
    Click on *learn more* for more details about your workspace.
    ::: 

3. Name your new workspace.  

4. Click on **Create Workspace** to run your workspace in a notebook tab.

    :::note

    Download the required base image and libraries before running Data Science or Deep Learning workspaces.

    :::
 