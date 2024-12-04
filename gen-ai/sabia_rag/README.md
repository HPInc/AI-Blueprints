# SabIA - A RAG-based agent running locally

SabIA is a project developed by students Software Innovation Lab (LIS) of the PUC-RS University, in partnership with HP Brazil R&D. The project consists in the development of an assistant for Z by HP AI Studio, based on Retrieval Augmented Generation (RAG) running in a local machine.

In a RAG pipeline, a Knowledge Base is created, using a Vector database to store chunks of documents that might be relevant. In SabIA, this knowledge base is created with a specific chain (LoaderChain), that can be run before or in parallel with the main inference service.

In the inference chain (question answering), the question typed by the user is used twofold. First, it is used to retrieve the relevant chunks of documentation from the Vector Dataset. Then, the question and the chunks are used again, together with the history of previous interactions, to create a prompt. This prompt is then sent to the LLM, so that an output answering the question can be produced.

Currently, SabIA uses ChromaDB as the vector Database, considering a 4b quantized version of Llama model for both the embedding and the LLM.

## Setup

Differently of other experiments run in AI studio, SabIA does not need to run on notebooks - it can be run directly from the Python code. For that, one needs to setup the project accordingly:
 * Create a new project
 * In the setup and documentation tab, setup the GitHub repo:
   * https://github.azc.ext.hp.com/phoenix/ds-experiments/ (for internal use with HP credentials)
   * https://github.com/passarel/aistudio-ds-experiments (Public repo for external use)
 * Download the model from 
   * HP Sharepoint: https://hp-my.sharepoint.com/:u:/p/rafael_borges/EdBEBYK1hmJJg_bugBZpYS8B_k7kCZB9LIWLmNLMSvQ0oA?e=mQkI2M
   * Add the downloaded model to a local folder, and add the folder as a dataset for the project (name it Llama7b)
 * Create a new NeMo Workspace
   * Run the workspace
   * Verify, in the Jupyter tab, if the code and the model are available on file browser on the left 
   * Inside Jupyter, start a new terminal
   * Enter the folder of the SabIA project:
     * cd .../ds-experiments/SabIA
   * Reduced version of SabIA can run as a console application with the following command:
	 * python app.py
   * Full version can be deployed as a model service, by running:
     * python app_deploy.py
	 * After that finishes, one can look in MLFlow on the monitoring tab to make sure the experiment and the run are logged, and the model is registered
	 * A new service can be started in the Model Deployment tab, by selecting the 
   * Both files can be opened inside jupyter to show how the app was implemented
   
	
   
 
