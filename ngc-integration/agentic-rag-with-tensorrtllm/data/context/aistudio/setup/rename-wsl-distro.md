---
title:  'Renaming a WSL Distribution'
sidebar_position: 4
---
# Renaming a WSL Distibution

AI Studio requires its own WSL distribution named "*Phoenix*". If our installer has directed you to these instructions, then you probably have a distribution that needs to be renamed before you can finish installing AI Studio.

### To rename your conflicting distro:

1. Terminate all instances of WSL.

2. From the Start menu, find and open the ***Registry Editor***.

3. Follow this path: 

    HKEY_CURRENT_USER\SOFTWARE\Microsoft\Windows\CurrentVersion\Lxss

4. Look at the value under *Distribution Name* to locate the folder you need to rename.

5. Right-click ***Distribution Name*** and select **modify**. Choose any value except "*Phoenix*", then click **OK** to save your changes.

6. Open a terminal window and run this command:
    ```powershell
	wsl –l
    ```

7. Review the output to ensure the distro was properly renamed.

After editing the conflicting distro name, rerun *AIStudioSetup.msi* to complete installation.