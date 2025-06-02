---
title:  'Add User to AI Studio Group'
sidebar_position: 5
---
# Add a User to the AI Studio Group

On Ubuntu, AI Studio requires each OS user to be a member of a specific group called "aistudio". If you've been directed to these instructions, then you probably need to add your own username to this group to use AI Studio.

To add a user to the AI Studio group:

1. Run the following command, entering your password if prompted. 
    ```bash
    sudo usermod -a -G aistudio $USER
    ```
    :::tip

    If you want to add a user other than the user you are currently logged in as, replace `$USER` with the relevant username.

    :::

2. Restart your computer so that the change can take effect.


Now that you are a member of the group, start AI Studio.
