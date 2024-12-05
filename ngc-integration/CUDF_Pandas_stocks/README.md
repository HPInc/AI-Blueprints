# Description
  In this project, we provide notebooks to compare execution time of dataset operations running on traditional Pandas (CPU) and in NVidia's CUDF Pandas extension, which is accelerated on GPU. This exemple is presented in two different ways
  * **original-example**: Example as created by NVidia, runs all the evaluation inside the Notebook code (including downloading data and restarting Kernel to activate extension)
  * **Separated Notebooks**: Uses differenty sized datasets from Data-Fabric, running in different notebooks (one for CPU, other for GPU), and logging the results with MLFlow

# Resources
## Recommended workspaces
  To run this experiment, we recommend to use either Rapids Base or Rapids Notebooks from NGC Catalog

## Datasets
  * In the **original-example** notebook, the data is downloaded directly from inside the code in NVidia Collab/GCP
  * For the experiments in separate notebooks, data can temporarily be found in the public s3 folder *s3://dsp-demo-bucket/rapids-data*, in AWS Zone us-west-2
  * For HP users, wanting to download the data locally, they can be found on https://hp.sharepoint.com/:f:/t/HPDataSciencePlatform/Emkn5C4s5a5PgROMPJAWAMEBOukgg8qfiRJO4ISg7gEYbA?e=hrcz9w

#
  