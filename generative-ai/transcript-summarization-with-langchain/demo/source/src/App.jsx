import { useState, useEffect } from 'react';
import { Card } from '@veneer/core';
import { IconInfo } from '@veneer/core';
import { Toggle } from '@veneer/core';
import { Button } from '@veneer/core';
import { TextArea } from '@veneer/core';
import { FilePicker } from '@veneer/core';
import * as pdfjsLib from 'pdfjs-dist';
import mammoth from 'mammoth';
import iconImage from '../icon.ico';
import './App.css';

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).href;

function App() {
	// State for file upload and processing
	const [file, setFile] = useState(null);
	const [fileText, setFileText] = useState("");
	const [fileName, setFileName] = useState("");
	const [fileType, setFileType] = useState("");
	
	// State for API response
	const [summaryResponse, setSummaryResponse] = useState("");
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);
	
	// State for UI display toggles
	const [showDetails, setShowDetails] = useState(false);
	const [expandedSection, setExpandedSection] = useState(null);
	
	/**
	 * Toggle expanded view for original text section
	 */
	async function toggleOriginalTextInfo() {
		if (expandedSection !== "original-text") {
			setExpandedSection("original-text");
		} else {
			setExpandedSection(null);
		}
	}
	
	/**
	 * Toggle expanded view for summarization section
	 */
	async function toggleSummaryInfo() {
		if (expandedSection !== "summary-info") {
			setExpandedSection("summary-info");
		} else {
			setExpandedSection(null);
		}
	}
	
	/**
	 * Toggle information about the black box mode
	 */
	async function toggleBlackBoxInfo() {
		setShowDetails(!showDetails);
	}
	
	/**
	 * Handle file selection
	 * @param {Event} e - The file input change event
	 */
	async function handleFileChange(e) {
		const selectedFile = e.target.files[0];
		if (!selectedFile) return;
		
		setFile(selectedFile);
		setFileName(selectedFile.name);
		setFileType(selectedFile.type);
		setError(null);
		
		try {
			const text = await extractTextFromFile(selectedFile);
			setFileText(text);
		} catch (err) {
			console.error("Error extracting text from file:", err);
			setError("Failed to extract text from the file. Please try a different format.");
			setFileText("");
		}
	}
	
	/**
	 * Extract text content from various file formats
	 * @param {File} file - The file to extract text from
	 * @returns {Promise<string>} - The extracted text
	 */
	async function extractTextFromFile(file) {
		try {
			// Simple text extraction based on file type
			if (file.type === "text/plain") {
				return await file.text();
			} else if (file.type === "application/pdf") {
				const arrayBuffer = await file.arrayBuffer();
				const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

				let extractedText = '';
				for (let i = 1; i <= pdf.numPages; i++) {
					const page = await pdf.getPage(i);
					const textContent = await page.getTextContent();
					const pageText = textContent.items.map(item => item.str).join(' ');
					extractedText += `Page ${i}:\n${pageText}\n\n`;
				}
				return extractedText || "PDF text extraction complete. No text found or PDF contains images.";
			} else if (file.type === "application/msword" || 
					file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
				const arrayBuffer = await file.arrayBuffer();
				const result = await mammoth.extractRawText({ arrayBuffer });
				return result.value || "No text could be extracted from the document.";
			} else {
				throw new Error("Unsupported file format");
			}
		} catch (error) {
			console.error("Error extracting text:", error);
			throw new Error(`Failed to extract text: ${error.message}`);
		}
	}
	
	/**
	 * Submit the extracted text to the API for summarization
	 */
	async function submitForSummarization() {
		if (!fileText) {
			setError("Please upload a file first.");
			return;
		}
		
		setLoading(true);
		setError(null);
		
		try {
			const requestBody = {
				inputs: {
					text: [fileText]
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
			}

			const jsonResponse = await response.json();

			if (jsonResponse.predictions) {
				if (Array.isArray(jsonResponse.predictions) && jsonResponse.predictions.length > 0) {
					const firstPrediction = jsonResponse.predictions[0];
					
					if (firstPrediction.summary) {
						setSummaryResponse(firstPrediction.summary);
					} else {
						setSummaryResponse("No summary data found in model response.");
					}
				} 
				// Handle direct object case
				else if (jsonResponse.predictions.summary) {
					setSummaryResponse(jsonResponse.predictions.summary);
				}
				else {
					setSummaryResponse("No summary provided by the model.");
				}
			} else {
				setSummaryResponse("Invalid response format from model.");
			}
		} catch (error) {
			console.error("Error when calling the API:", error);
			setError(`Failed to get summary: ${error.message}`);
		} finally {
			setLoading(false);
		}
	}
	
	const showOriginalTextInfo = expandedSection === "original-text";
	const showSummaryInfo = expandedSection === "summary-info";
	
	return (
		<div>
			<div className="header">
				<div className="header-logo">
					<img src={iconImage} width="150px" height="150px" alt="Transcript Summarization Logo" /> 
				</div>
				<div className='title-info'>
					<div className="header-title">
						<h3 className='title'>Text Summarization with AI Studio</h3>
					</div>
					<div className="header-description">
						<p>Extract and summarize information from text documents</p>
					</div>
				</div>
			</div>
			
			{/* File Upload Card */}
			<Card className="file-upload-card"
				border="outlined"
				content={
					<div className="outer-padding">
						<h4>Upload Document</h4>
						<p>Select a text file, PDF, or Word document to summarize.</p>
						<div className="file-input-container">
							<FilePicker
								id="file-picker"
								accept=".txt,.pdf,.doc,.docx,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
								error={!!error}
								helperText={error}
								onChange={handleFileChange}
								className="file-input"
								onClear={() => {
									setFile(null);
									setFileText("");
									setError(null);
								}}
							/>
						</div>
						<div className="input-control input-buttons">
							<div className='input-toggle'>
								<Toggle className="detail-toggle" label="Show Original Text" onChange={setShowDetails} />
							</div>
							<Button 
								className="submit-button" 
								onClick={submitForSummarization}
								disabled={!fileText || loading}
							>
								{loading ? "Processing..." : "Summarize"}
							</Button>
						</div>
					</div>
				}
			/>
			
			{/* Main content area with either detailed view or simplified view */}
			{showDetails ? (
				<Card className="white-box"
					border="outlined"
					content={
						<div className="main-detail-box">
							{/* Original Text Card */}
							<Card className={`text-module-card ${showOriginalTextInfo ? "card-expanded" : "card-not-expanded"}`}
								border="outlined"
								content={
									<div className='outer-padding'>
										<div className='title-with-icon'>
											<h5>Original Document Text</h5>
											<div className='title-with-icon-icon'>
												{showOriginalTextInfo ? 
													<IconInfo size={24} onClick={toggleOriginalTextInfo} filled /> :
													<IconInfo size={24} onClick={toggleOriginalTextInfo} />
												}
											</div>
										</div>
										<div className="text-info">
											{showOriginalTextInfo && 
												<p>
													This is the extracted text from your uploaded document. 
													The quality of extraction depends on the document format and structure.
												</p>
											}
											<TextArea 
												className="original-text-area" 
												id="original-text"
												label="Extracted text:"
												value={fileText} 
												readOnly
												separateLabel
												onChange={() => {}} 
											/>
										</div>
									</div>
								}
							/>
							
							{/* Summary Output Card */}
							<Card className={`text-module-card ${showSummaryInfo ? "card-expanded" : "card-not-expanded"}`}
								border="outlined"
								content={
									<div className='outer-padding'>
										<div className='title-with-icon'>
											<h5>Summarized Output</h5>
											<div className='title-with-icon-icon'>
												{showSummaryInfo ? 
													<IconInfo size={24} onClick={toggleSummaryInfo} filled /> :
													<IconInfo size={24} onClick={toggleSummaryInfo} />
												}
											</div>
										</div>
										<div className="summary-info">
											{showSummaryInfo &&	
												<p>
													This is the AI-generated summary of your document. The model extracts key information 
													while preserving the essential meaning and context.
												</p>
											}
											<TextArea 
												className="summary-text-area" 
												id="summary-text"
												label="Generated Summary:"
												value={summaryResponse} 
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
				/>
			) : (
				<div>
					<Card className="black-box"
						border="outlined"
						content={
							<div>
								<div className='title-with-icon' style={{display:"flex", justifyContent:"center"}}>
									<h4>Document Summarization</h4>
									<div className='title-with-icon-icon'>
										{showDetails ? 
											<IconInfo size={24} onClick={toggleBlackBoxInfo} filled /> :
											<IconInfo size={24} onClick={toggleBlackBoxInfo} />
										}
									</div>
								</div>
								<div>
									{error && <p className="error-message">{error}</p>}
								</div>
								<div className='outer-padding'>
									<TextArea 
										className="output-external-area" 
										id="output-text"
										value={summaryResponse || "Upload a document and click 'Summarize' to generate a summary."} 
										readOnly
										onChange={() => {}} 
									/>
								</div>
							</div>
						}
					/>
				</div>
			)}
		</div>
	);
}

export default App;
