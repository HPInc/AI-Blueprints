---
title:  'P2P Data Sharing'
sidebar_position: 5
---
# P2P Data Sharing

AI Studio uses peer-to-peer networking to synchronize data in the background in these three circumstances: 

1. Data recorded by MLflow

    :::note

    This data includes model parameters, experiment results and model data.

    :::

2. Data output to Tensorboard 

3. Assets in the shared directory (see below) 

Synchronization operates asynchronously in the background while AI Studio is running.  All members of an account contribute to a mesh p2p network.  As files are added to paths monitored for synchronization, they are synchronized across all account members. 

:::note

Two or more account members must be online to sync. 

:::

Synchronized paths (Windows) 

- %LOCALAPPDATA%\HP\AIStudio\[account ID]\projects\[project ID]\mlflow 

- %LOCALAPPDATA%\HP\AIStudio\[account ID]\projects\[project ID]\tensorboard 

- %LOCALAPPDATA%\HP\AIStudio\[account ID]\projects\[project ID]\shared 

  

Synchronized paths (Linux XDG_STATE_HOME unset) 

- $HOME/.state/hp/aistudio/[account ID]/projects/[project ID]/mlflow 

- $HOME/.state/hp/aistudio/[account ID]/projects/[project ID]/tensorboard 

- $HOME/.state/hp/aistudio/[account ID]/projects/[project ID]/shared 

  

Synchronized paths (Linux XDG_STATE_HOME set) 

- $XDG_STATE_HOME/hp/aistudio/[account ID]/projects/[project ID]/mlflow 

- $XDG_STATE_HOME/hp/aistudio/[account ID]/projects/[project ID]/tensorboard 

- $XDG_STATE_HOME/hp/aistudio/[account ID]/projects/[project ID]/shared 

  

:::warning

Simultaneously editing files in shared paths (e.g. storing a jupyter notebook in a shared tree to collaborate) is not advised. HP recommends using AI Studio’s git integration features to develop code collaboratively. 

:::