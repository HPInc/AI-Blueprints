
document.getElementById("text_translation").addEventListener("click", translateText);
document.getElementById("audio_translation").addEventListener("click", translateAudio);

function translateAudio(){
    reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onloadend = function () {
        let base64String = reader.result;
        audioB64 = base64String.substr(base64String.indexOf(',') + 1);
       	response = sendRequest(null, audioB64);

        
        
        document.getElementById("spanish_text").innerHTML = response["predictions"]["translated_text"];
        document.getElementById("english_text").innerHTML = response["predictions"]["original_text"];
        audioURL = "data:image/png;base64," + response["predictions"]["translated_serialized_audio"];
        fetch(audioURL).then(res => res.blob()).then(res => output_audio.src = window.URL.createObjectURL(res))
        
    } 
}

function translateText(){
	englishText = document.getElementById("english_text").value;
	document.getElementById("spanish_text").innerHTML = sendRequest(englishText);
}


function sendRequest(inputText, audioB64){
	useAudio = (audioB64 != null)
	if (!useAudio){
		audioB64 = ""
	}
	requestBody = {
		"inputs": {
			"source_text": [inputText],
			"source_serialized_audio": [audioB64]
    	},
	"params": {
		"use_audio": useAudio,
        "save_audio_input": useAudio,
        "save_audio_output": useAudio
		}
	}
	request = new XMLHttpRequest();
	request.open("POST", "/invocations", false)
	request.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
	request.send(JSON.stringify(requestBody))
	return JSON.parse(request.response)
}


	