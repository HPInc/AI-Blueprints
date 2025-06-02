import { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import { Card } from '@veneer/core';
import { IconInfo } from '@veneer/core';
import { Toggle } from '@veneer/core';
import { Button } from '@veneer/core';
import { TextArea } from '@veneer/core'
import './App.css'

function App() {
	const [inputQuery, setInputQuery] = useState("");
	const [modelResponse, setModelResponse] = useState("");
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
	
	async function submitInput() {
	  console.log("Send:", inputQuery);
	  try {
		// setLoading(true);
		const requestBody = {
		  inputs: {
			question: [inputQuery]
		  },
		  params: {}
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
  
		const predictions = jsonResponse.predictions;
		setModelResponse(predictions.output || "No output provided by the model.");
		setInteractionHistory(predictions.history || "No output History.");
		setPromptEngineering(predictions.prompt || "No prompt retrieved.");
		setVectorChunks(predictions.chunks.join('\n') || "No chunks retrieved.");
  
		
		// setLoading(false);
	  } catch (error) {
		console.error("Error when calling the request:", error);
		// setLoading(false);
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
						<Button className="submit-button" onClick={submitInput}>
							Submit input
						</Button>
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
				{/* <Card className="output-external-card"
					border="outlined"
					headers
					content={
						<div>
							<TextArea className="output-external-area" id="output-text"
								value={modelResponse || ""} 
								readOnly
								onChange={() => {}} 
							/>
						</div>
					}
				/> */}
			</div>
		}
	</div>
  )
}

export default App




