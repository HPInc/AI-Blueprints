---
title:  'Creating a New Job'
sidebar_position: 1
---

# Creating a New Job

###  To create a new job: 

1. Click on **Create New Job**. 

    :::note
    You need to clone the git repository you want to run the job on if you haven’t already before you can continue. 
    :::

2. Name, date, and schedule your job. 

    :::note
    Enter time in ddmmyyy format. 
    :::

3. Choose a workspace to run the job on. 

4. Select a Branch / Tag or Commit from the dropdown menu. 

    :::note
    These outputs depend on the git repository associated with the job.
    ::: 

5. Enter the script you want your job to execute in the script path field. 

    :::note
    You must select a Branch / Tag or Commit to make this field editable.
    ::: 

6. Add any arguments you wish to apply to the job in the ***Script Arguments*** window. 

7. Click on **Review and Create** to run a quick check to make sure you’ve satisfied all the fields before moving on to the next step. 

8. Choose the GPU(s) you want to allocate to the Job. 

9. Add the variables you want to set in the job runner at runtime. 

10. Review your job, then click **Create Job** to save your configuration. 

 