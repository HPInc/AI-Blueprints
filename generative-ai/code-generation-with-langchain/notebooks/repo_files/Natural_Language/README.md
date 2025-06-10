![alt](../images/NLP.png)
# NLP Experiments

Experiments that involve training and inference of deep learning models with text-related tasks such as summarization, text generation, and entity classification. Recommended for showcasing features related to GPU, Monitor Table, and Published Services. Provides models for inference via Hugging Face.

---
#### [Question answering with BERT ](bert_qa/)
In this experiment, we will train one of the most classic language models, BERT, using a question and answer dataset called SQuAD. The model will be able to answer simple questions based on contexts like this:

    📄 "Imperialism has greatly shaped the contemporary world. It has also allowed for the rapid spread of technologies and ideas. The term imperialism has been applied to Western (and Japanese) political and economic dominance especially in Asia and Africa in the 19th and 20th centuries."

    ❔"Imperialism is responsible for the rapid spread of what?"

    🤖 "technologies and ideas"

| Needed Resources     |              |
|--------------|--------------        |
| Recommended workspaces         | Large, Deep Learning   |
| GPU enabled          | Yes           |
| Minimum GPU memory   | 4GB    |

| Integrations     |
|--------------|
| Published Services (Swagger)|
| MLFlow            |       
---

----------

#### [Shakespeare text generation with RNN](text_generation/)
In this experiment, we built a recurrent neural network that will be able to predict the next character given a starting word. The training vocabulary is based on Shakespeare's Sonnet 1, and therefore the network will be able to generate texts similar to the author's, given an initial word:

    ❤️ "Love is"

    🤖 "Love is no man. The streams of heart and sender than the way of traitor..."

| Needed Resources     |              |
|--------------|--------------        |
| Recommended workspaces         | Large, Deep Learning   |
| GPU enabled          | Yes           |
| Minimum GPU memory   | 4GB    |

| Integrations     |
|--------------|
| Published Services (Swagger)|
| MLFlow            |       
---