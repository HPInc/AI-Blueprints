import mlflow
from mlflow import MlflowClient
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import ParamSchema, ParamSpec
from mlflow.models import ModelSignature
from transformers import pipeline
import torch
import json
import os

dir = os.path.dirname(__file__)
demoPath = os.path.join(dir, 'demo')
print("demoPath:", demoPath)

class DistilBERTModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.MODEL = "morgana-rodrigues/bert_qa"
        self.qa = pipeline('question-answering',
            model=self.MODEL,
            device=-1 #-1 means running on CPU
            )        
    def _preprocess(self, inputs):
        context = inputs['context'][0]
        question = inputs['question'][0]
        print("pre processing", context,question)
        return context, question
        
    def load_context(self, context):
        self.model = pipeline(
            'question-answering',
             model=context.artifacts["model"],
             device=-1
        )
        
    def predict(self, context, model_input, params):
        in_ctx, question = self._preprocess(model_input)
        output = self.model(context=in_ctx, question=question)
        return output

    @classmethod
    def log_model(cls, model_name, trainer = None, pipeline = None, demo_folder="demo"): #eg (model, '', 'my_model')
        input_schema = Schema(
            [
                ColSpec("string", "context"),
                ColSpec("string", "question"),
            ]
        )
        output_schema = Schema(
            [
                ColSpec("string", "answer")
            ]
        )
        
        params_schema = ParamSchema(
            [
                ParamSpec("show_score", "boolean", False)
            ]
        )
      
        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=params_schema)
        if trainer is not None:
            trainer.save_model(model_name)
        elif pipeline is not None:
            pipeline.save_pretrained(model_name)
             
        requirements = [
            "transformers==4.37.0",
            "numpy==1.24.3",
            "torch==2.0.0",
            "tqdm==4.65.0",
        ]
        mlflow.pyfunc.log_model(
            model_name,
            python_model=cls(),
            artifacts={"model": model_name, "demo": demo_folder},
            signature=signature,
            pip_requirements=requirements
        )
       
    def execute(self, **kwargs):
        mlflow.set_experiment(experiment_name='BERT for Q&A')

        with mlflow.start_run(run_name='BERT_QA') as run:
            print(f"Run's Artifact URI: {run.info.artifact_uri}")
            DistilBERTModel.log_model(model_name='BERT_QA', pipeline=self.qa, demo_folder=demoPath)
            mlflow.register_model(model_uri = f"runs:/{run.info.run_id}/BERT_QA", name='BERT_QA')

        client = mlflow.MlflowClient()
        model_metadata = client.get_latest_versions("BERT_QA", stages=["None"])
        latest_model_version = model_metadata[0].version
        print(latest_model_version, mlflow.models.get_model_info(f"models:/BERT_QA/{latest_model_version}").signature)

        model = mlflow.pyfunc.load_model(model_uri=f"models:/BERT_QA/{latest_model_version}")
        context = "Marta is mother of John and Amanda"
        question = "what is the name of Marta's daugther?"
        model.predict({"context": [context], "question":[question]})

if __name__ == "__main__":
    DistilBERTModel().execute()
    