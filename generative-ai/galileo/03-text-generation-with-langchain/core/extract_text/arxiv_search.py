import requests
import xml.etree.ElementTree as ET
import logging
from langchain_community.document_loaders import PyMuPDFLoader

class ArxivSearcher:
    """
    Searches arXiv for articles based on a query and extracts the text from the associated PDFs.
    """

    def __init__(self, query: str, max_results: int = 1, logging_enabled: bool = False):
        """
        Initializes the ArxivSearcher with a query and optional logging.

        Args:
            query (str): Search term.
            max_results (int): Number of articles to retrieve.
            logging_enabled (bool): Enable detailed logging output.
        """
        self.query = query
        self.max_results = max_results
        self.api_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
        self.namespace = '{http://www.w3.org/2005/Atom}'
        self.logger = logging.getLogger(__name__)

        if logging_enabled:
            logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        else:
            logging.disable(logging.CRITICAL)

    def search_and_extract(self):
        """
        Searches arXiv and extracts text from article PDFs.

        Returns:
            list[dict]: List of articles with 'title' and 'text'.
        """
        response = requests.get(self.api_url)
        papers = []

        if response.status_code != 200:
            self.logger.error("Failed to access the arXiv API.")
            return papers

        root = ET.fromstring(response.content)

        for entry in root.findall(f'{self.namespace}entry'):
            title = entry.find(f'{self.namespace}title').text
            pdf_url = self._extract_pdf_url(entry)

            if pdf_url:
                pdf_path = self._format_pdf_filename(title)
                if self._download_pdf(pdf_url, pdf_path):
                    text = self._extract_text_from_pdf(pdf_path)
                    papers.append({'title': title.strip(), 'text': text})
                    self.logger.info(f"Extracted text from '{title.strip()}':\n{text[:300]}...\n")
                else:
                    self.logger.warning(f"Failed to download PDF for '{title.strip()}'.")
            else:
                arxiv_url = entry.find(f'{self.namespace}id').text
                self.logger.warning(f"No PDF found for '{title.strip()}'. See online: {arxiv_url}")

        return papers

    def _extract_pdf_url(self, entry):
        for link in entry.findall(f'{self.namespace}link'):
            if link.attrib.get('type') == 'application/pdf':
                return link.attrib['href']
        return None

    def _format_pdf_filename(self, title):
        safe_title = title[:50].strip().replace(' ', '_')
        return f"temp_{safe_title}.pdf"

    def _download_pdf(self, pdf_url, output_path):
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            self.logger.error(f"Error downloading PDF: {response.status_code}")
            return False

    def _extract_text_from_pdf(self, pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])