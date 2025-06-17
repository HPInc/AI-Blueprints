---
title:  'Connecting Data to a Project'
sidebar_position: 2
---

# Connecting Data to a Project

Connect data to your projects using local storage or your AWS credentials to access remote AWS storage

### To connect data to a new project:

1. When you finish adding project details, connect your data by simply selecting **Add** beside the asset(s) you want to add. Then, click **Create Project**. 

2. If there are no available assets, click **Create new Asset** to connect to a new one. 

    - Name and describe your asset, then specify your connection source (i.e., local, AWS).  

        - **Local**: Enter the path or browse your local directory to locate the asset, then click **Save**. 

        - **AWS**: Enter the S3 URI associated with your asset, the bucket type, region (optional for private assets), and description. Then, click **Create Asset**.

            :::tip

            Public AWS S3 buckets don’t require credentials and authentication.

            :::
        - **Azure Blob Storage**: Enter the Blob URI associated with your asset, the resource type, and an asset description. Then, click **Create Asset**.

### To connect data to an existing project:

1. From the project’s *Overview* page, select the **Assets** tab and navigate to the ***Project Assets*** window. 

2. Click **Catalog** to connect an existing asset to the project from the *Assets Catalog*. 

3. If the asset you wish to add is already listed, click **Add** to upload it into your project.  
    :::tip
    Click on the more options icon to edit the asset before you add it to your project.
    ::: 

4. If your desired asset hasn’t been connected, click **New Asset** to connect to a new one. 
    - Name and describe your asset, then specify your asset type and connection source (i.e., local, AWS, or Azure Blob Storage). 

        - **Local**: Enter the path or navigate the file browser to locate the asset, then click **Save**. 

        - **AWS**: Enter the S3 URI associated with your asset, the bucket type, region (optional for private assets), and description. Then, click **Create Asset**. 

        - **Azure Blob Storage**: Enter the Blob URI associated with your asset, the resource type, and an asset description. Then, click **Create Asset**. 