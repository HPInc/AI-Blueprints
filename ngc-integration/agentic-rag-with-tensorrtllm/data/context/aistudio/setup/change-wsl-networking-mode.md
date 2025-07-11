---
title:  'Changing WSL Networking Mode'
sidebar_position: 4
---
# Changing WSL's Networking Mode

AI Studio requires WSL's networking mode to be "NAT". If our installer directed you to these instructions, then you probably have changed your networking mode to "mirrored" or something else that needs to be changed to "NAT" before you can use AI Studio.

> 💡 For background, see Microsoft’s documentation on [WSL configuration settings](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#configuration-settings-for-wslconfig) and [mirrored mode networking](https://learn.microsoft.com/en-us/windows/wsl/networking#mirrored-mode-networking).


### To change your WSL networking mode:

1. Terminate all instances of WSL

2. Press Windows + R, type `%userprofile%`, and press Enter. Then, open the `.wslconfig` file in a text editor

3. Find the `networkingMode` key and change its value to `nat`, like this: `networkingMode=nat`

4. Open a terminal and run the following to force WSL to restart and apply the changes:
    ```powershell
    wsl --shutdown
    ```

After you've updated the networking mode and restarted WSL, relaunch AI Studio.