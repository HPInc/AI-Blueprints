---
title:  'Installing & Uninstalling AI Studio'
sidebar_position: 2
---
# Installing & Uninstalling AI Studio

## Installation
### To install AI Studio on Windows:

1. Download and run the AI Studio installer <a href="/downloads">here</a>. If you haven't enabled virtualization, an error message will prompt you to do so before continuing. 

2. Complete the wizard to finish installing AI Studio.

3. Click on the AI Studio desktop icon to start using AI Studio.

    :::note

    The application automatically checks for a compatible existing distro and leverages it to install and configure the necessary images. If none exists, it will add one for you on your first start up.

    :::

4. On your first start up, you’ll have the option to choose where to download the WSL distro. Click **Next** to install the AI Studio WSL distro to the default folder or click the folder icon and use the file browser to select a different folder.

   ![Distro selection modal](/img/distro-select.png)

:::tip

If git is not already installed on your machine, the app will guide you to do so. Features that depend on git are disabled in the app for users who choose not to install. 

:::

### To install AI Studio on Ubuntu:

1. Download and run the AI Studio installer <a href="/downloads">here</a>.

2. After finishing the download, open the terminal in the same directory as your downloaded file and run the command:

    `sudo apt install ./AIStudioSetup[app version].deb`

    :::note

    Apt may give a warning that says "*Download was performed unsandboxed as root*". This is normal and does not mean the install failed.

    :::

3. Restart your Ubuntu machine to make sure your user is added to the AI Studio permissions group.

You can open AI Studio by searching for it in the applications menu, or by typing `AIStudio` in the terminal.

:::note

The first time you start the app on Ubuntu, AI Studio will ask for permission to set up nvidia-container-toolkit if you don't already have it. 

:::

## Uninstallation

### To uninstall AI Studio on Windows: 

Use either of the following methods to uninstall AI Studio on Windows: 

**Using the Installer**: 

1. Run the *AIStudioSetup.msi* file to open the installer. 

2. Click **Next**, then click **Remove** to delete AI Studio from your computer. 

**Using System Settings**: 

1. From System Settings, find and select **Add or Remove Programs**. 

2. Search for AI Studio in the ***Apps and Features*** section or scroll to locate it manually. 

3. Click the options icon, then select **Uninstall**. 

### To uninstall AI Studio on Ubuntu:

From the terminal, run the command:

`sudo apt remove hpaistudio`

:::note

This will **not** delete your user files. Uninstalling AI Studio on Ubuntu leaves your user files intact.

:::
