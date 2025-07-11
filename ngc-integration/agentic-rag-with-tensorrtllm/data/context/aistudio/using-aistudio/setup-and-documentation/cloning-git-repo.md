---
title:  'Cloning a GitHub Repository'
sidebar_position: 1
---
# Cloning a GitHub Repository  

Use AI Studio to share data and collaborate with your teammates without changing the workflows your team is most familiar with. You can connect AI Studio to your GitHub account by cloning a repository into your workspace to collaboratively manage your code. 

### To clone your repository: 

 1. From your Project's Overview page, select the ***Project Setup*** tab, navigate to setup, then click **Clone GitHub Repository**.
 
 2. Type your GitHub Repository URL in the space provided.
 
 3. Specify the local folder where you want to download the cloned repository.
      :::warning
      Your file path should **not** include spaces. If the repo fails to clone, check your local file path to ensure it doesn’t contain any forbidden characters.
      :::
 
 4. Click **Add GitHub Repository** to clone the repository.

    :::note
     When you add your cloned repository, AI Studio redirects you to the GitHub authentication page. Allow GitHub to access the necessary data from your account, then return to the application to start using data on the cloned repository. 
     :::

 5. Restart your workspace if it’s open to apply changes.
 
    :::tip
    You can access and manipulate data from cloned repositories with the notebook tab that appears when you run your workspace, exactly like you would with a local or remote AWS connection.
    :::

 ### To push code changes to the repository:

 1. Run the workspace to open a notebook tab.
 
 2. Click on the git action icon or use the Git menu. 

 3. Select **Initialize a Repository**, then click **Yes** to begin initialization. 

Now you can now use the Jupyter extension to push and pull code updates to the associated repository. 