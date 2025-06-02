---
title:  'Adding Custom Libraries'
sidebar_position: 2
---
# Adding Custom Libraries 

For larger projects that need more nuanced customization and powerful compute, you can set up your workspace with custom libraries. Custom workspaces sync with collaborators in shared projects, so teams know they’re using workspaces in the same environments.

:::tip
Customize base AI Studio templates by uploading a requirements.txt file. 
:::

### To choose your environment: 

1. Choose a project and navigate to *My Workspaces* on the project’s home page. 

2. Click on **New Workspace**. 

3. Choose the workspace environment that is most compatible with your machine(s) and project requirements.

    :::tip
    Check out our guide on supported [NGC containers](/docs/aistudio/using-aistudio/workspaces/base-images#ngc-catalog-containers).
    :::

4. (*Recommended*) Download the Deep Learning or Data Science base image before adding those libraries to the workspace.

    :::note
    If images are not downloaded during workspace creation, they must be downloaded before starting the workspace. 
    :::

5. Name your workspace and specify your project's graphical requirements. 

6. Click **Add Custom Libraries** to add any custom Python libraries you want to use in your workspace. 
    :::note
    Workspaces with custom libraries have two kernels available to run a jupyter notebook in (*conda env:base or conda env:aistudio*).

    When you launch a workspace, the default kernel is set to *conda env:base* for all custom workspaces. Users must validate their custom libraries, then manually change the kernel to *conda env:aistudio* to use custom libraries with the workspace.
    ::: 

7. Paste applicable libraries from a [compatible base image](/docs/aistudio/using-aistudio/workspaces/base-images), then click **Validate libraries** to test your package.

    :::note
    You can use your own libraries, but they may be incompatible with the libraries necessary to run the required base images. Validating libraries is optional but assists in troubleshooting.
    :::

8. Click **Create Workspace** to save your new workspace. 

9. Click on the play icon in your workspace tile to instantiate your environment.

    :::warning
    The blue progress bar indicates that the application is still building the container for your workspace. This process should take about 5 minutes, depending on your device’s capabilities. **DO NOT** navigate away from the window or select another workspace until the progress bar vanishes.
    :::

AI Studio projects can support more than one workspace to accommodate projects that require an ensemble of models to complete your team’s work.