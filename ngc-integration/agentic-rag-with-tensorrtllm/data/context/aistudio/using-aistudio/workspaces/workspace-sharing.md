---
title:  'Workspace Sharing'
sidebar_position: 4
---
# Workspace Sharing

Share assets and file updates with collaborators in real-time with AI Studio’s workspace sharing features.

:::tip
View the documentation on Networking and Sync Protocols to ensure your team’s system configurations are set up to support AI Studio. 
:::

### To share data from a notebook tab:

1. Select the shared folder from the notebook file browser. 

2. Click **Upload files** and select the files you wish to share. 

3. Click **Open** to add the files to your shared folder, then click **Save**. 
 
    :::note
    
    Avoid simultaneously modifying files in the datasync tree (i.e. user A and user B editing the same notebook at the same time).
    
    :::

Files placed in a project’s shared folder leverage Syncthing to synchronize online collaborators’ workspaces and must sync completely before other users can access them. Syncing larger assets can take several minutes to appear for your collaborators. Synchronization only occurs for a user’s last active account – other accounts don't sync in the background.
 