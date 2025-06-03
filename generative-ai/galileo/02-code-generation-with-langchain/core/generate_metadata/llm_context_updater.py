import httpcore
import logging
from tqdm import tqdm

class LLMContextUpdater:
    def __init__(self, llm_chain, prompt_template, verbose=False, print_prompt=False, overwrite=True):
        """
        :param llm_chain: LLM chain object with an .invoke() method
        :param prompt_template: PromptTemplate used to render the final prompt
        :param verbose: If True, enable logging output
        :param print_prompt: If True, print the formatted prompt
        :param overwrite: If True, always overwrite context even if it exists
        """
        self.llm_chain = llm_chain
        self.prompt_template = prompt_template
        self.print_prompt = print_prompt
        self.overwrite = overwrite

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)  

    def update(self, data_structure):
        updated_structure = []

        for item in tqdm(data_structure, desc="Updating Contexts"):
            try:
                code = item['code']
                filename = item['filename']
                context = item.get('context', '')

                if context and not self.overwrite:
                    self.logger.debug(f"Skipping context for: {filename}")
                    updated_structure.append(item)
                    continue

                inputs = {
                    "code": code,
                    "filename": filename,  
                    "context": context
                }

                rendered_prompt = self.prompt_template.format(**inputs)

                if self.print_prompt:
                    self.logger.debug(f"\nPrompt for file {filename}:\n{rendered_prompt}\n{'=' * 60}")

                response = self.llm_chain.invoke(inputs)
                item['context'] = response.strip()

                self.logger.debug(f"[LOG] Context updated for: {filename}")  

            except httpcore.ConnectError as e:
                self.logger.error(f"Connection error on {filename}: {str(e)}")
            except httpcore.ProtocolError as e:
                self.logger.error(f"Protocol error on {filename}: {str(e)}")
            except Exception as e:
                self.logger.error(f"General error on {filename}: {str(e)}")

            updated_structure.append(item)

        return updated_structure
