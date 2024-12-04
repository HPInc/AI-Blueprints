import React from "react";
import ReactAudioPlayer from "react-audio-player";

type PlayProps = {
  outputAudio: string;
};

const GeneratedAudio: React.FC<PlayProps> = ({ outputAudio }: PlayProps) => {
  return (
    <div className="p-10">
      <h3 className="title-small">Generated Audio Translation</h3>
      <div className="pt-10">
        <div>
          <ReactAudioPlayer className="w-full" src={outputAudio} controls />
        </div>
      </div>
    </div>
  );
};

export default GeneratedAudio;
