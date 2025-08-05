---
sidebar_label: 'Limitations & Known Issues'
sidebar_position: 6
---

# Limitations & Known Issues

## 1.45.2

### Known Issue – Pulling Models into your Project

When creating a project in AI Studio, you will need to create a workspace. After creating the workspace, the assets will appear but they don't download automatically. This is because while the workspace is starting up, it does not pull down the assets. 

---
To download the assets, manually click the download icon for each asset. This requires stopping the workspace, downloading the assets, and then restarting the workspace to pull the model into your project.

Follow these steps as a workaround:

1.	Create a workspace (Note that assets will appear but won't be automatically downloaded).
2.	Take note that while the workspace is starting up, it does not pull down the assets.
3.	Manually click the download icon for each asset.
4.	Stop the workspace.
5.	Download the assets.
6.	Restart the workspace to pull the model into your project.

### Error loading DistroEnablers

#### Issue description: 
After upgrading, users may see an error titled "Error loading" or "Error loading DistroEnablers" while starting the app. This error occurs on versions below 1.45.5. On version 1.45.5, the error is fixed, but users may encounter the error once on this patched version.

---

#### Workarounds
Relaunch the application after upgrading to the fixed version or uninstall and reinstall the patched version is another solution. Both options fix the issue. 
Take note that uninstalling and reinstalling without upgrading the application to the fixed version is a temporary solution, and the user will encounter the bug again when they perform another upgrade.



## 1.35.8
- Users who can’t update their workspace(s) to bypass future PIP Installs can add the PIP install on their script explicitly as a workaround. This is especially true for Ubuntu packages installed via apt-get. 

    :::warning
    This process still takes time to execute the command each time the script is run.
    ::: 

    Users who are only installing new python packages can alternatively create a new workspace and add all of the python packages when customizing it to alleviate the need to install them at runtime. 

- You can’t add models to a running workspace, so add any models you might find useful before you run it. 

- When Cloning a GitHub Repository, your file path should **not** include spaces. If the repo fails to clone, check your local file path to ensure it doesn’t contain any forbidden characters. 

- When a workstation hibernates, all networking connections are closed (including the one AIS has open to Jupyter Lab). The TCP connection is reestablished automatically, so it's safe for you to simply dismiss the dialog and resume working.

## 1.31.1
- Your session may expire when you run AI Studio for an extended period of time without restarting the application. If this occurs, your experience may include the following:

    1. You're unable to view the full list of team members from the Account tab.

    2. You recieve a toast explaining that AI Studio is unable to fetch content.

    3. Experiment Data fails to sync among team members.
 
    If your issue is related to an expired session, restarting AI Studio and logging back in when prompted should fix it.

## 1.18.23
- Newly added team members may have trouble accessing files in the shared folder their first time logging in to AI Studio. After the new user accepts their invite, both the sender and user trying to access the shared folder must restart AI Studio for the documents to sync properly.


## 1.18.22
- If the app freezes when trying to add a team member, you may need to restart the app. After reopening AI Studio, the added team member should appear under *Pending* in Team Settings.


## 1.18.21
- WSL may time out if AI Studio is left open in the background too long, interfering with expected workspace behavior. Log out and close the application, then reopen AI Studio and log back in to reset WSL and restore normal functionality. 

- For some users, ML flow and Tensorboard take longer than expected to sync across collaborators who are working together in a shared folder. This is, in part, due to a bug with the Syncthing GUI that causes a delay before indicating that the machines are in sync with each other.