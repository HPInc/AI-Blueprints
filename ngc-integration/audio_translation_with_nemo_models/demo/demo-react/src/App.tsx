// @ts-nocheck
// import AudioRecorder from "simple-audio-recorder";

import React, { useEffect, useState } from "react";
import "./App.css";
import logo from "./assets/Z-HP.png";
// import useAudioRecorder from "./utils/useSimpleAudioRecorder";
import TranslateAudio from "./TranslateAudio";
import TranslateText from "./TranslateText";
import { Select } from "@veneer/core";
import { sendRequest } from "./utils/sendRequest";
// import useAudioRecorder from "./utils/useAudioRecorder";
// import { useAudioRecorder } from "@sarafhbk/react-audio-recorder";
// import { blobToBase64  } from 'file64';

// import useAudioRecorder from "./utils/useAudioRecorder";
// import useRawWavRecorder from "./utils/useRawWavRecorder";
// import useAudioRecorder from "./utils/useAudioRecorder"; // Import the hook
import useASR, { ConfigType } from "react-asr";

const blobToBase64 = (blob) => {
  return new Promise((resolve, _) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.readAsDataURL(blob);
  });
};

const App: React.FC = () => {
  // const { startRecording, stopRecording, clearRecording, record, audioBlob } =
  //   useRecorder({
  //     sampleRate: 44100, // Sample rate example
  //   });

  const [loading, setLoading] = useState(false);
  const [englishText, setEnglishText] = useState<string>("");
  const [spanishText, setSpanishText] = useState<string>("");
  const [outputAudio, setOutputAudio] = useState<string>("");

  const {
    isRecording,
    recordedBlob: audioBlob,
    startRecording,
    stopRecording,
  } = useASR<ConfigType>({
    MIN_DECIBELS: -80,
    DETECTION_INTERVAL: 100,
    SILENCE_DURATION: 2000,
    LOG_ENABLED: false,
  });

  const [value, setValue] = useState<string[]>(["audio"]);
  const onChange = (selectedOption: any): void => {
    setEnglishText("");
    setSpanishText("");
    setValue([selectedOption.value]);
  };

  // as soon as the recorded file is available, start processing it.
  useEffect(() => {
    const fetchData = async (): Promise<void> => {
      try {
        setLoading(true);
        console.log(audioBlob);
        const audioBase64 = await blobToBase64(audioBlob);
        console.log(audioBase64);
        const val = audioBase64.substr(audioBase64.indexOf(",") + 1);
        console.log(val);
        const json = await sendRequest(null, val);
        console.log(json);
        setEnglishText(json.predictions.original_text);
        setSpanishText(json.predictions.translated_text);
        const newAudioUrl = `data:image/png;base64,${json["predictions"]["translated_serialized_audio"]}`;
        const res = await fetch(newAudioUrl);
        const newBlob = await res.blob();
        const audio = window.URL.createObjectURL(newBlob);
        setOutputAudio(audio);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching audio:", error);
        setLoading(false);
      }
    };
    if (audioBlob) {
      fetchData();
    }
  }, [audioBlob]);

  const onClickText = (): void => {
    const run = async (): Promise<void> => {
      try {
        setLoading(true);
        const response: any = await sendRequest(englishText, null);
        setSpanishText(response.predictions.translated_text);
        setLoading(false);
      } catch (error) {
        console.error("Error translating text:", error);
        setLoading(false);
      }
    };
    run();
  };

  return (
    <>
      <div className="flex size-full flex-col">
        <div className="relative flex justify-center py-4">
          <img alt="HP Z Logo" className="w-16" src={logo} />
          <div className="absolute right-[25%] top-0 flex h-full flex-col justify-center">
            <Select
              className="test"
              clearIcon={false}
              defaultValue={["audio"]}
              helperTextVisibility="auto"
              id="select-usage"
              label="Select"
              options={[
                { value: "audio", label: "Translate Audio" },
                { value: "text", label: "Translate Text" },
              ]}
              placement="bottom"
              value={value}
              onChange={onChange}
            />
          </div>
        </div>
        {value[0] === "audio" ? (
          <>
            <TranslateAudio
              audioBlob={audioBlob || ""}
              englishText={englishText}
              isRecording={isRecording}
              loading={loading}
              outputAudio={outputAudio}
              spanishText={spanishText}
              startRecording={startRecording}
              stopRecording={stopRecording}
            />
          </>
        ) : (
          <TranslateText
            englishText={englishText}
            loading={loading}
            setEnglishText={setEnglishText}
            spanishText={spanishText}
            onClickText={onClickText}
          />
        )}
      </div>
    </>
  );
};

export default App;
