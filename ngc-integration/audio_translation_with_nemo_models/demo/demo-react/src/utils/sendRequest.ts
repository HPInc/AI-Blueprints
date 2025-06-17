/* eslint-disable  @typescript-eslint/no-explicit-any */

// request to demo api
const sendRequest = async (
  inputText: string | null,
  audioB64: string | null,
): Promise<any> => {
  const useAudio = audioB64 !== null;
  if (!useAudio) {
    audioB64 = "";
  }
  const requestBody = {
    inputs: {
      source_text: [inputText],
      source_serialized_audio: [audioB64],
    },
    params: {
      use_audio: useAudio,
      save_audio_input: useAudio,
      save_audio_output: useAudio,
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
    // if (!response.ok) {
    //   throw new Error("Network response was not ok");
    // }
    const jsonResponse = await response.json();
    return jsonResponse;
  } catch (error) {
    console.error("There was a problem with the fetch operation:", error);
    throw error;
  }
};

export { sendRequest };
