# Notebook templates for integrating AI Studio and Galileo

In this folder, we bring a series of use cases to illustrate the integration between AI Studio and Galileo, often through the use of LangChain to orchestrate the language pipelines and allow the logging of metrics into Galileo. The following examples are available.

## Chatbot (chatbot-with-langchain.ipynb)

In this simpler example, we implement a basic chatbot assistant to AI Studio, by means of a basic RAG pipeline. In this pipeline, we load information from a document (AIStudio documentation) into a vector database, then use this information to answer questions about AI Studio through proper prompting and use of LLMs. In the example, we illustrate different ways to load the model (locally and from cloud), and also illustrate how to use Galileo's callbacks to log information from the LangChain modules

## Summarization (summarization-with-langchain.ipynb)

For this use case, we extend the basic scenario to include more complex pre-processing of the input. In our scenario, we break an original transcript (which might be too long) into smaller topics (chunks with semantic relevance). A chain is then built to summarize the chunks in parallel, then joining them into a single summary in the end.

Also in this example, we illustrate how to work with
* Personalized runs from Galileo (using EvaluateRuns)
* Personalized Metrics that runs locally (using CustomScorers)

## Code Generation (code-generation-with-langchain.ipynb)

This use case illustrate an example where the user access a git repository to serve as code reference. Some level of code understanding is used to index the available code from the repository. Based on this reference, the code generator uses in-cell prompts from the user in order to generate code in new notebook cells. 

## Text Generation (code-generation-with-langchain.ipynb)

This use case shows a process to search for a scientific paper in ArXiv, than generating a presentation based on the content of this paper.