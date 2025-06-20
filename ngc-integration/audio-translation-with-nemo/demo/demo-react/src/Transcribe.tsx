import React from "react";
import { TextArea } from "@veneer/core";

type TranscribeProps = {
  title: string;
  value: string;
  placeholder: string;
  className?: string;
  readOnly?: boolean;
  setEnglishText?: (val: string) => void;
};

const Transcribe: React.FC<TranscribeProps> = ({
  title,
  value,
  placeholder,
  className = "",
  readOnly = false,
  setEnglishText,
}: TranscribeProps) => {
  return (
    <div className={`h-full ${className} p-[30px]`}>
      <h3 className="title-small">{title}</h3>
      <div className="h-[80%]">
        <TextArea
          className="text-area-fix h-full"
          defaultValue={value}
          height="200px"
          label=""
          placeholder={placeholder}
          readOnly={readOnly}
          onChange={(val) => setEnglishText(val)}
        />
      </div>
    </div>
  );
};

export default Transcribe;
