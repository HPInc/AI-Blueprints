import React from "react";
import { Card } from "@veneer/core";
import Transcribe from "./Transcribe";
import Record from "./Record";
import GeneratedAudio from "./GeneratedAudio";

type TranslateAudioProps = {
  englishText: string;
  spanishText: string;
  loading: boolean;
  outputAudio: string;
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<void>;
  audioBlob: any;
  isRecording: boolean;
};

const TranslateAudio: React.FC<TranslateAudioProps> = ({
  englishText,
  loading,
  spanishText,
  startRecording,
  stopRecording,
  audioBlob,
  isRecording,
  outputAudio,
}: TranslateAudioProps) => {
  return (
    <div className="flex size-full flex-row justify-evenly p-8 pt-0">
      <div className="h-full w-1/3 pl-8 pr-4">
        <Card
          border="outlined"
          className="relative flex h-full flex-col justify-center"
          content={
            <>
              <Record
                audioBlob={audioBlob}
                isRecording={isRecording}
                startRecording={startRecording}
                stopRecording={stopRecording}
              />
              {/* {audioURL && <div><button>Submit</button></div>} */}
              <div
                className={`absolute bottom-[20%] w-full ${loading ? "visible" : "invisible"}`}
              >
                <div className="spinner"></div>
              </div>
            </>
          }
        />
      </div>
      <div className="flex h-full w-1/3 flex-col px-4">
        <Card
          border="outlined"
          className="h-full"
          content={
            <div className="flex h-full flex-col justify-evenly">
              <Transcribe
                placeholder="Transcription in English"
                title="Generated English Transcription"
                value={englishText}
                readOnly
              />
              <Transcribe
                placeholder="Translation in Spanish"
                title="Generated Spanish Translation"
                value={spanishText}
                readOnly
              />
            </div>
          }
        />
      </div>
      <div className="h-full w-1/3 pl-4 pr-8">
        <Card
          border="outlined"
          className="flex h-full flex-col justify-center"
          content={<GeneratedAudio outputAudio={outputAudio} />}
        />
      </div>
    </div>
  );
};

export default TranslateAudio;
