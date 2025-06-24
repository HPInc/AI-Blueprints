import React, { useState } from "react";
import ReactAudioPlayer from "react-audio-player";

import recorder from "./assets/record.svg";
import arrow from "./assets/arrow.svg";

type RecordProps = {
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<void>;
  audioBlob: any;
  isRecording: boolean;
};

const Record: React.FC<RecordProps> = ({
  startRecording,
  stopRecording,
  audioBlob,
  isRecording,
}: RecordProps) => {

  const handleClick = async () => {
    if (isRecording) {
      await stopRecording();
    } else {
      await startRecording();
    }
  };

  const src = audioBlob ? URL.createObjectURL(audioBlob) : ""

  return (
    <div className="p-10">
      <h3 className="title-small">Record</h3>
      <div>
        <div className="mb-2 flex justify-center pl-10 pr-10 pt-5 pb-5">
          <div
            className={isRecording ? "recording" : ""}
            id="recorder"
            onClick={handleClick}
          >
            <img
              alt="recording icon"
              draggable="false"
              id="record"
              src={recorder}
            />
            <img alt="arrow" draggable="false" id="arrow" src={arrow} />
          </div>
        </div>
        <div>
          <ReactAudioPlayer className="w-full" src={src} controls />
        </div>
      </div>
    </div>
  );
};

export default Record;
