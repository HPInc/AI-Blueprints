import React, { FC } from "react";
import { TextArea as Input } from "@veneer/core";

type Props = {
  title: string;
  value: string;
  placeholder: string;
  className?: string;
  readOnly?: boolean;
  onChange?: (value: string) => void;
  type?: "number" | "text";
};

const TextArea: FC<Props> = ({
  onChange = () => null,
  title,
  value,
  placeholder,
  className = "",
  readOnly = false,
  type = "text",
}) => {
  return (
    <div className={`h-full ${className} p-[30px]`}>
      <h3 className="title-small">{title}</h3>
      <div className="">
        <Input
          className="text-area-fix h-full"
          defaultValue={value}
          height="200px"
          label=""
          placeholder={placeholder}
          readOnly={readOnly}
          type={type}
          onChange={onChange}
        />
      </div>
    </div>
  );
};

export default TextArea;
