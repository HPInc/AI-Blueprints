// @ts-nocheck
// import AudioRecorder from "simple-audio-recorder";

import React, { useEffect, useState } from "react";
import "./App.css";
import logo from "./assets/Z-HP.png";
import { sendRequest } from "./utils/sendRequest";
import Content from "./Content";

const App: React.FC = () => {

  return (
    <>
      <div className="flex size-full flex-col">
        <div className="relative flex justify-center py-4">
          <img alt="HP Z Logo" className="w-16" src={logo} />
        </div>
        <div>
          <Content />
        </div>
      </div>
    </>
  );
};

export default App;
