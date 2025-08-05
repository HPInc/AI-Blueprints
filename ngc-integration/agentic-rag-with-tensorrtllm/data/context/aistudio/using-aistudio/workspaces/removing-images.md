---
title:  'Removing Expired Images'
sidebar_position: 8
---
# Removing Expired Images

AI Studio downloads new workspace images whenever new versions are created and a newer app version is installed, but does not automatically remove the old images. If you wish to free up the disk space occupied by the old images, you can do so manually. 

### To manually remove images (Windows): 

**List images**

1. To view a list of the images on your device, use the command: 

    ```
    wsl -d phoenix -- sudo ctr -n phoenix i ls
    ```

2. Identify the images that display multiple versions and remove them with the following command below. 

**Remove images** 

1. To remove the images identified in the previous step, run the following command: 
    ```
    wsl -d phoenix -- sudo ctr -n phoenix i rm <unused image>
    ```

    :::tip

    Image names look like this: `public.ecr.aws/aistudio/prod/base-images/minjp:0.14.1`
    
    :::

### To manually remove images (Ubuntu): 

**List images**

1. To view a list of the images on your device, use the command:

    ```
    ctr -a /run/hp/aistudio/containerd/containerd.sock -n phoenix i ls
    ```

2. Identify the images that display multiple versions and remove them with the following command below. 

**Remove images**

To remove the images identified in the previous step, use the command: 
    ```
    ctr -a /run/hp/aistudio/containerd/containerd.sock -n phoenix i rm <unused image>
    ``` 
