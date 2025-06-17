import React, { FC } from "react";
import { TextArea } from "@veneer/core";

type Props = {
  title: string;
  value: string;
  placeholder: string;
  className?: string;
  readOnly?: boolean;
  onChange?: (value: string) => void;
};

const Inputs: FC<Props> = ({
  onChange = () => null,
  title,
  value,
  placeholder,
  className = "",
  readOnly = false,
}) => {
  return (
    <div className={`h-full ${className} p-[30px]`}>
      <h3 className="title-small">{title}</h3>
      <div className="">
        <TextArea
          className="text-area-fix h-full"
          defaultValue={value}
          height="200px"
          label=""
          placeholder={placeholder}
          readOnly={readOnly}
          onChange={onChange}
        />
      </div>
    </div>
  );
};

export default Inputs;
