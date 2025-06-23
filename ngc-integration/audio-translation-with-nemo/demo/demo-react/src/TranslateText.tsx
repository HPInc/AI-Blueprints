import React from "react";
import { Card, Button } from "@veneer/core";
import Transcribe from "./Transcribe";

type TranslateTextProps = {
  englishText: string;
  spanishText: string;
  loading: boolean;
  onClickText: () => void;
  setEnglishText: (val: string) => void;
};

const TranslateText: React.FC<TranslateTextProps> = ({
  englishText,
  loading,
  onClickText,
  spanishText,
  setEnglishText,
}: TranslateTextProps) => {
  return (
    <div className="flex size-full flex-row justify-evenly p-8 pt-0">
      <div className="h-full w-1/3 pl-8 pr-4">
        <Card
          border="outlined"
          className="relative flex h-full flex-col justify-center"
          content={
            <Transcribe
              placeholder="Text in English"
              title="Enter English Text"
              value={englishText}
              setEnglishText={setEnglishText}
            />
          }
        />
      </div>
      <div className="flex w-1/6 flex-col  justify-center">
        <Button
          appearance="primary"
          disabled={!englishText}
          trailingIcon={
            loading ? (
              <div className="ml-3">
                <div className="spinner" />
              </div>
            ) : (
              <div />
            )
          }
          onClick={onClickText}
        >
          {loading ? "Translating" : "Translate"}
        </Button>
      </div>
      <div className="h-full w-1/3 pl-4 pr-8">
        <Card
          border="outlined"
          className="flex h-full flex-col justify-center"
          content={
            <Transcribe
              placeholder="Translation in Spanish"
              title="Generated Spanish Translation"
              value={spanishText}
              readOnly
            />
          }
        />
      </div>
    </div>
  );
};

export default TranslateText;
