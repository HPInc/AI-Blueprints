import React, { FC, useState } from "react";
import { Card, Button } from "@veneer/core";
import Inputs from "./Inputs";

type ResponseType =
  | undefined
  | {
      predictions: {
        score: number;
        start: number;
        end: number;
        answer: string;
      };
    };

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

const Content: FC = () => {
  const [loading, setLoading] = useState(false);
  const [context, setContext] = useState<string>("");
  const [question, setQuestion] = useState<string>("");

  const [output, setOutput] = useState<ResponseType>(undefined);

  const submit = async (): Promise<void> => {
    setLoading(true);
    const requestBody = {
      inputs: {
        context: [context],
        question: [question],
      },
      params: {
        show_score: true,
      },
    };
    try {
      const response = await fetch("/invocations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json;charset=UTF-8",
        },
        body: JSON.stringify(requestBody),
      });
      const json = await response.json();
      setOutput(json);
      // await wait(2000);
      // setOutput({
      //   predictions: { score: 0.9, start: 0, end: 0, answer: "Hello" },
      // });
    } catch (error) {
      console.error("There was a problem with the fetch operation:", error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex size-full flex-row justify-evenly p-8 pt-0">
      <div className="flex h-full w-1/3 flex-col gap-20 pl-8 pr-4">
        <Card
          border="outlined"
          className="relative flex h-full flex-col justify-center"
          content={
            <Inputs
              placeholder="Context"
              title="Context"
              value={context}
              onChange={(value) => setContext(value)}
            />
          }
        />
        <Card
          border="outlined"
          className="relative flex h-full flex-col justify-center"
          content={
            <Inputs
              placeholder="Question"
              title="Question"
              value={question}
              onChange={(value) => setQuestion(value)}
            />
          }
        />
      </div>
      <div className="flex w-1/6 flex-col  justify-center">
        <Button
          appearance="primary"
          disabled={loading || context === "" || question === ""}
          trailingIcon={
            loading ? (
              <div className="ml-3">
                <div className="spinner" />
              </div>
            ) : (
              <div />
            )
          }
          onClick={submit}
        >
          {loading ? "Submitting" : "Submit"}
        </Button>
      </div>
      <div className="h-full w-1/3 pl-4 pr-8">
        <Card
          border="outlined"
          className="flex h-full flex-col justify-center"
          content={
            <div>
              <Inputs
                className="h-auto pb-0"
                placeholder="Output"
                title="Output"
                value={output?.predictions.answer || ""}
                readOnly
              />
              <div className="flex flex-row justify-center">
                <span
                  className={`${output !== undefined ? "visible" : "invisible"}`}
                >
                  Score: {output?.predictions.score}
                </span>
              </div>
            </div>
          }
        />
      </div>
    </div>
  );
};

export default Content;
