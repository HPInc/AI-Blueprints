import { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import { Card } from '@veneer/core';
import { IconInfo } from '@veneer/core';
import { Toggle } from '@veneer/core';
import { Button } from '@veneer/core';
import { TextArea } from '@veneer/core';
import { IconDocument } from '@veneer/core';
import { ConfirmationModal } from '@veneer/core';
import { IconTextAlignCenter } from '@veneer/core';
import { IconPlus } from '@veneer/core'

import './App.css'

function App() {
	const [inputQuery, setInputQuery] = useState("");
	const [modelResponse, setModelResponse] = useState("");
	const [isModalVisible, setIsModalVisible] = useState(false);
	const [isModalVisiblePDF, setIsModalVisiblePDF] = useState(false);
	const showModal = () => setIsModalVisible(true);
	const showModalPDF = () => setIsModalVisiblePDF(true);
	const [yourPrompt, setYourPrompt] = useState("");

  	const hideModal = () => setIsModalVisible(false);
	const hideModalPDF = () => setIsModalVisiblePDF(false);
	const [showPromptEdit, setShowPromptEdit] = useState(false);

	const [loading, setLoading] = useState(false);

	// const [loading, setLoading] = useState(false);
	const [interactionHistory, setInteractionHistory] = useState("");
	const [promptEngineering, setPromptEngineering] = useState(""); 
	const [vectorChunks, setVectorChunks] = useState("");
	const [showWhiteBox, setShowWhiteBox] = useState(false);
	// const [showVectorInfo, setShowVectorInfo] = useState(false);
	// const [showHistoryInfo, setShowHistoryInfo] = useState(false);
	// const [showPromptInfo, setShowPromptInfo] = useState(false);
	// const [showLlmInfo, setShowLlmInfo] = useState(false);
	const [expandedSection, setExpandedSection] = useState(null)
	const [showBlackBoxInfo, setShowBlackBoxInfo] = useState(false);
 
	async function toggleVectorInfo() {
		// setShowVectorInfo(!showVectorInfo);
		// setExpandedSection("vector-info")
		if (expandedSection !== "vector-info") {
			setExpandedSection("vector-info")
		} else {
			setExpandedSection(null)
		}
	}
	
	async function toggleHistoryInfo() {
		// setShowHistoryInfo(!showHistoryInfo);
		// setExpandedSection("history-info")
		if (expandedSection !== "history-info") {
			setExpandedSection("history-info")
		} else {
			setExpandedSection(null)
		}

	}
	
	async function togglePromptInfo() {
		// setShowPromptInfo(!showPromptInfo);
		// setExpandedSection("prompt-info")
		if (expandedSection !== "prompt-info") {
			setExpandedSection("prompt-info")
		} else {
			setExpandedSection(null)
		}
	}
	
	async function toggleLlmInfo() {
		if (expandedSection !== "llm-info") {
			setExpandedSection("llm-info")
		} else {
			setExpandedSection(null)
		}
	}
	
	async function toggleBlackBoxInfo() {
		setShowBlackBoxInfo(!showBlackBoxInfo);
	}
	
// Submit Input
// async function submitPDF(event) {
// 	console.log("Chamando submitPDF");
//     const file = event.target.files[0];
//     if (file) {
 
        
       
//         const reader = new FileReader();

//         reader.onload = async (loadEvent) => {
//             const base64String = loadEvent.target.result;
//             const base64FormattedString = base64String.replace('data:application/pdf;base64,', '');

           


//             try {
             
//                 const requestBody = {
//                     inputs: {
//                         pdfBase64: base64FormattedString
//                     },
//                     params: {}
//                 };

//                 const response = await fetch("/invocations", {
//                     method: "POST",
//                     headers: {
//                         "Content-Type": "application/json;charset=UTF-8",
//                     },
//                     body: JSON.stringify(requestBody),
//                 });

//                 if (!response.ok) {
//                     throw new Error(`HTTP error! status: ${response.status}`);
//                 }

//                 const jsonResponse = await response.json();
//                 console.log("JSON Response:", jsonResponse);

                
//                 setVectorChunks(jsonResponse.predictions.chunks.join('\n') || "No chunks retrieved.");
                
//             } catch (error) {
//                 console.error("Error when sending the PDF:", error);
//             } finally {
//                 // setLoading(false);
//             }
//         };

//         reader.onerror = (error) => {
//             console.error('Error reading the PDF file:', error);
//         };

        
//         reader.readAsDataURL(file);
//     }
// }

// async function insertPDF() {
// 	console.log("Calling insertPD");
//     const fileInput = document.getElementById('pdfUploader');
  

//     fileInput.onchange = submitPDF;
  
  
//     fileInput.click();
// }

// event input
async function submitPDF(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();

        reader.onload = async (loadEvent) => {
            const base64String = loadEvent.target.result;
            const base64FormattedString = base64String.split(',')[1]; 

            console.log("PDF em base64:", base64FormattedString);

            try {
                const requestBody = {
                    inputs: {
                        query: [""], 
                        prompt: [""],
                        document: [base64FormattedString]
                    },
                    params: {
                        add_pdf: true, 
                        get_prompt: false, 
                        set_prompt: false
                    }
                };

                const response = await fetch("/invocations", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json;charset=UTF-8",
                    },
                    body: JSON.stringify(requestBody),
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const jsonResponse = await response.json();
                console.log("JSON Response:", jsonResponse);

                if (jsonResponse && jsonResponse.predictions) {
                    const { chunks } = jsonResponse.predictions;
                    setVectorChunks(Array.isArray(chunks) ? chunks.join('\n') : "PDF sent successfully");
                } else {
                    console.error('Unexpected JSON response format:', jsonResponse);
                }
            } catch (error) {
                console.error("Error when sending the PDF:", error);
            }
        };

        reader.onerror = (error) => {
            console.error('Error reading the PDF file:', error);
        };

        reader.readAsDataURL(file);
    }
}

async function insertPDF() {
    const fileInput = document.getElementById('pdfUploader');
    fileInput.onchange = submitPDF;
    fileInput.click();
}




//submit input
// async function submitCustomPrompt() {
// 	console.log("Sending custom prompt:", yourPrompt);
// 	try {
// 	  const requestBody = {
// 		inputs: {
// 		  custom_prompt: yourPrompt
// 		},
// 		params: {}
// 	  };
// 	  const response = await fetch("/invocations", {
// 		method: "POST",
// 		headers: {
// 		  "Content-Type": "application/json;charset=UTF-8",
// 		},
// 		body: JSON.stringify(requestBody),
// 	  });
  
// 	  if (!response.ok) {
// 		throw new Error(`HTTP error! status: ${response.status}`);
// 	  };
  
// 	  const jsonResponse = await response.json();
// 	  console.log("JSON Response:", jsonResponse);
  
	
// 	  setPromptEngineering(jsonResponse.predictions.prompt || "No prompt retrieved.");
  
// 	} catch (error) {
// 	  console.error("Error when sending the custom prompt:", error);
// 	} finally {
// 	  // setLoading(false); 
// 	}
//   }

// event input
async function submitInput() {
    console.log("Send:", inputQuery);
    try {
        // setLoading(true);
        const requestBody = {
            inputs: {
                query: [inputQuery], 
                prompt: [""], 
                document: [""] 
            },
			params: {
				add_pdf: false, 
				get_prompt: false, 
				set_prompt: false
			}
        };
        const response = await fetch("/invocations", {
            method: "POST",
            headers: {
                "Content-Type": "application/json;charset=UTF-8",
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        };

        const jsonResponse = await response.json();
        console.log("JSON Response:", jsonResponse);

        
        const { output, history, prompt, chunks } = jsonResponse.predictions;

        
        setModelResponse(output || "No output provided by the model.");
        setInteractionHistory(history || "No interaction history.");
        setPromptEngineering(prompt || "No prompt retrieved.");

        
        setVectorChunks(Array.isArray(chunks) ? chunks.join('\n') : "No chunks retrieved.");

        
    } catch (error) {
        console.error("Error when calling the request:", error);
        // setLoading(false);
    }
}



	  
//	async function submitInput() {
//		console.log("Send:", inputQuery);
//		setLoading(true); 
//		try {
//		  const requestBody = {
//			inputs: {
//			  question: [inputQuery]
//			},
//			params: {}
//		  };
//		  const response = await fetch("/invocations", {
//			method: "POST",
//			headers: {
//			  "Content-Type": "application/json;charset=UTF-8",
//			},
//			body: JSON.stringify(requestBody),
//		  });
//	  
//		  if (!response.ok) {
//			throw new Error(`HTTP error! status: ${response.status}`);
//		  }
//	  
//		  const jsonResponse = await response.json();
//		  console.log("JSON Response:", jsonResponse);
//	  
//		 
//		  setModelResponse(jsonResponse.predictions.output || "No output provided by the model.");
//		  setInteractionHistory(jsonResponse.predictions.history || "No output History.");
//		  setPromptEngineering(jsonResponse.predictions.prompt || "No prompt retrieved.");
//		  setVectorChunks(jsonResponse.predictions.chunks.join('\n') || "No chunks retrieved.");
//		} catch (error) {
//		  console.error("Error when calling the request:", error);
//		} finally {
//		  setLoading(false); 
//		}
//	  }





async function submitCustomPrompt(yourPrompt) {
    console.log("Sending custom prompt:", yourPrompt);
    try {
        const requestBody = {
            inputs: {
                query: [""], 
                document: [""], 
                prompt: [yourPrompt] 
            },
            params: {
                add_pdf: false,
                get_prompt: false,
                set_prompt: true
            }
        };

        const response = await fetch("/invocations", {
            method: "POST",
            headers: {
                "Content-Type": "application/json;charset=UTF-8",
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const jsonResponse = await response.json();
        console.log("JSON Response:", jsonResponse);

        setPromptEngineering(jsonResponse.predictions.prompt || "Prompt send");
		
    } catch (error) {
        console.error("Error when sending the custom prompt:", error);
    } finally {
        setLoading(false);
    }
}

async function fetchPrompt() {
    try {
        const requestBody = {
            inputs: {
                query: [""],
                document: [""],
                prompt: [""] 
            },
            params: {
                add_pdf: false,
                get_prompt: true,  
                set_prompt: false
            }
        };

        const response = await fetch("/invocations", {
            method: "POST",
            headers: {
                "Content-Type": "application/json;charset=UTF-8",
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const jsonResponse = await response.json();
        console.log("Fetched Prompt:", jsonResponse);

        
        setYourPrompt(jsonResponse.predictions.prompt_template || "");
        showModal();  
    } catch (error) {
        console.error("Error fetching the custom prompt:", error);
    }
}


  const showVectorInfo = expandedSection === "vector-info"
  const showHistoryInfo = expandedSection === "history-info"
  const showPromptInfo = expandedSection === "prompt-info"
  const showLlmInfo = expandedSection === "llm-info"

  return (
    <div>
		<div className="header">
			<div className="header-logo">
				<img src="icon.ico" width="100px" height="100px" /> 
			</div>
			<div className='title-info'>
				<div className="header-title">
					<h3 className='title'>SabIA: Developing RAG in AI Studio</h3>
				</div>
				<div className="header-lis">
					<p>Developed in partnership with LIS/PUCRS</p>
				</div>
			</div>
		</div>
		<Card className="input-query-card"
			border="outlined"
			content={
				<div className="input-text-area-container outer-padding">
					<TextArea className="input-text-area" 
						id="input-text"
						label="Input query:"
						placeholder=""
						separateLabel
						value={inputQuery}
						onChange={(e) => setInputQuery(e)}
					/>
					<div className="input-control input-query-buttons">
						<div className='input-query-toggle'>
							<Toggle className="white-box-toggle" label="White Box" onChange={setShowWhiteBox} />
						</div>
						{loading ? (
							<Button disabled>Load...</Button> 
							) : (
							<Button className="submit-button" onClick={submitInput}>Submit input</Button>
							)}

					</div>
					
				</div>
			}
		/>
		{ showWhiteBox ? 
			<Card className="white-box"
				border="outlined"
				content={
					<div className="main-white-box">
						<Card className={`vector-module-card ${showVectorInfo ? "card-expanded" : "card-not-expanded"}`}
							// styles={{width: showVectorInfo ? "75%" : "auto"}}
							border="outlined"
							content={
								<div className='outer-padding'>
									<div className='title-with-icon'>
										<h5> ChromaDB vector database </h5>
										<div className='title-with-icon-icon'>
											{ showVectorInfo ? 
												<IconInfo size={24} onClick={toggleVectorInfo} filled />:
												<IconInfo size={24} onClick={toggleVectorInfo} />
											}
										</div>
										<div className='inset-pdf-button'>
											<IconPlus size={24}  onClick={showModalPDF}>
												Insert PDF
											</IconPlus>
											{isModalVisiblePDF && (
										<div className='modal-prompt'>
											<ConfirmationModal
											show={isModalVisiblePDF}
											confirmButtonLabel="Submit PDF"
											closeButtonLabel="Cancel"
         									title="Insert PDF"
											onConfirm={() =>{
												insertPDF();
												hideModalPDF();}}
											onClose={hideModalPDF}

										> 
										<p>To create chunks, please upload a PDF document.</p>
										</ConfirmationModal>
										</div>
										)}
											<input type="file" id="pdfUploader" style={{display: 'none'}} accept="application/pdf" onChange={submitPDF} />
										</div>
										
									</div>
									<div className="vector-info">
										{ showVectorInfo && 
											<p>
												A set of documents is broken into chunks and loaded into the DB. When the user inputs a query, relevant chunks of documents are retrieved, according to the relevance (semantic similarity) to the query.
											</p>
										}
										<div className="input-text-area-container"> 
											<TextArea className="vector-text-area" 
												id="vector-text"
												value={vectorChunks || ""} 
												label="Retrieved chunks:"
												readOnly
												separateLabel
												onChange={() => {}} 
											/>
										</div>
									</div>							
								</div>
							}
						/>
						<Card className={`vector-module-card ${showHistoryInfo || showPromptInfo ? "card-expanded" : "card-not-expanded"}`}
							border="outlined"
							// styles={{width: showHistoryInfo ? "75%" : "auto"}}
							content={
								<div className='outer-padding middle-container'>
							<div>
								<div className='title-with-icon'>
									<h5> Interaction history </h5>
									<div className='title-with-icon-icon'>
										{ showHistoryInfo ? 
											<IconInfo size={24} onClick={toggleHistoryInfo} filled />:
											<IconInfo size={24} onClick={toggleHistoryInfo} />
										}
									</div>
									<div className='add-prompt'></div>
								</div>
								<div className="input-text-area-container">	
									<TextArea className="input-text-area" 
										id="input-text"
										value={interactionHistory || ""}
										readOnly
										onChange={() => {}}
									/>
								</div>	
							</div>
							<div>
									<div className='title-with-icon'>
									
										<h5> Prompt engineering </h5>
										<div className='title-with-icon-icon'>
											{ showPromptInfo ? 
												<IconInfo size={24} onClick={togglePromptInfo} filled />:
												<IconInfo size={24} onClick={togglePromptInfo} />
											}
										</div>
										<IconPlus className="button-edit-prompt" onClick={fetchPrompt}>
											Prompt
										 </IconPlus>
										 {isModalVisible && (
											<div className='modal-prompt'>
												<ConfirmationModal
													show={isModalVisible}
													confirmButtonLabel="Submit"
													closeButtonLabel="Cancel"
													title="Prompt Navigator"
													onConfirm={() => {
														submitCustomPrompt(yourPrompt);
														hideModal();
													}} 
													onClose={hideModal}
												>
													<TextArea 
													className='text-prompt'
														id="resize-text-area" 
														label="Your Prompt:"
														resize="vertical"
														separateLabel
														value={yourPrompt}
														//onChange={(e) => setYourPrompt(e.target.value)}
														onChange={(e) => {
													
															const value = e.target ? e.target.value : e;
															setYourPrompt(value);
														}}
		
													/>
												</ConfirmationModal>
											</div>	
										)}


									</div>
									<div className="prompt-info">
										{ showPromptInfo &&
											<p>
												In this module, the retrieved chunks of documents and the history of previous interactions are used to build a prompt to be presented to the model. This prompt is engineered to give information about the agent and the desired behaviour.
											</p>
										}
										<TextArea className="prompt-text-area" 
											id="prompt-text"
											label="Generated prompt:"
											value={promptEngineering || ""} 
											readOnly
											separateLabel
											onChange={() => {}} 
										/>
	
		
				
									</div>
								</div>
						</div>
							}
						/>
						<Card className={`output-module-card ${showLlmInfo ? "card-expanded" : "card-not-expanded"}`}
							border="outlined"
							content={
								<div className='outer-padding'>
									<div className='title-with-icon'>
										<h5>Local LLM (Llama7b)</h5>
										<div className='title-with-icon-icon'>
											{ showLlmInfo ? 
												<IconInfo size={24} onClick={toggleLlmInfo} filled />:
												<IconInfo size={24} onClick={toggleLlmInfo} />
											}
										</div>
									</div>
									<div className="output-info">
										{ showLlmInfo &&	
											<p>
												This is where the actual large language model is run. Based on the input prompt, the Llama7b is loaded locally, and generates an output one word at a time.
											</p>
										}
										<TextArea className="output-text-area" id="output-text"
											value={modelResponse || ""} 
											label="Agent Output:" 
											readOnly
											separateLabel
											onChange={() => {}} 
										/>
									</div>
								</div>
							}
						/>
					</div>
				}				
			/> :
			<div>
				<Card className="black-box"
					border="outlined"
					content={
						<div>
							<div className='title-with-icon' style={{display:"flex", justifyContent:"center"}}>
								<h4> SabIA: RAG-based agent, running entirely on device </h4>
								<div className='title-with-icon-icon'>
									{ showBlackBoxInfo ? 
										<IconInfo size={24} onClick={toggleBlackBoxInfo} filled />:
										<IconInfo size={24} onClick={toggleBlackBoxInfo} />
									}
								</div>
							</div>
							<div>
								{ showBlackBoxInfo &&
									<p>
										This box represents an RAG-based agent, which uses an LLM running locally, and enhanced by a retrieval module that feeds relevant information from a knowledge base.
									</p>
								}
							</div>
							<div className='outer-padding'>
							<TextArea className="output-external-area" id="output-text"
								value={modelResponse || ""} 
								readOnly
								onChange={() => {}} 
							/>
						</div>
						</div>
					}
				/>
			</div>
		}
	</div>
  )
}

export default App
