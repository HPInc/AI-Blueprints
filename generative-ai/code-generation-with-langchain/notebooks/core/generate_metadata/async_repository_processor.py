"""
Asynchronous Repository Processor

This module provides functionality to process GitHub repositories asynchronously,
allowing the API to respond immediately without blocking while repository
processing continues in the background.
"""

import threading
import logging
import time
import hashlib
import concurrent.futures
from typing import Dict, Any, Optional, Callable, List

from .repository_status_tracker import RepositoryStatusTracker, ProcessingStatus

class AsyncRepositoryProcessor:
    """
    Handles asynchronous processing of GitHub repositories.
    
    This class:
    - Coordinates with RepositoryStatusTracker to track processing status
    - Launches repository processing in background threads
    - Ensures the API remains responsive regardless of repository size
    - Caches processed repositories to avoid redundant work
    
    Usage:
        processor = AsyncRepositoryProcessor()
        # Start processing in background
        processor.process_repository_async("https://github.com/username/repo")
        # Get current status
        status = processor.get_repository_status("https://github.com/username/repo")
    """
    
    def __init__(self, 
                 repository_extractor_class=None, 
                 llm_context_updater_class=None,
                 max_workers: int = 3,
                 status_tracker: Optional[RepositoryStatusTracker] = None):
        """
        Initialize the async repository processor.
        
        Args:
            repository_extractor_class: The class to use for repository extraction
            llm_context_updater_class: The class to use for context generation
            max_workers: Maximum number of concurrent background processing threads
            status_tracker: Optional status tracker instance (creates one if not provided)
        """
        self.repository_extractor_class = repository_extractor_class
        self.llm_context_updater_class = llm_context_updater_class
        self.max_workers = max_workers
        self.status_tracker = status_tracker or RepositoryStatusTracker()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.active_futures: Dict[str, concurrent.futures.Future] = {}
        self._lock = threading.RLock()
        
    def _generate_repo_id(self, repo_url: str, branch: str = "main") -> str:
        """
        Generate a unique identifier for a repository.
        
        Args:
            repo_url: URL of the GitHub repository
            branch: Branch name (default: "main")
            
        Returns:
            A unique hash string for the repository
        """
        # Create a unique hash from the URL and branch
        repo_key = f"{repo_url}:{branch}"
        return hashlib.md5(repo_key.encode()).hexdigest()
        
    def get_repository_status(self, repo_url: str, branch: str = "main") -> Dict[str, Any]:
        """
        Get the current status of repository processing.
        
        Args:
            repo_url: URL of the GitHub repository
            branch: Branch name (default: "main")
            
        Returns:
            A status dictionary with current progress information
        """
        repo_id = self._generate_repo_id(repo_url, branch)
        status = self.status_tracker.get_status(repo_id)
        
        if status is None:
            return {
                "status": ProcessingStatus.NOT_STARTED.value,
                "progress": 0,
                "message": "Repository processing not started"
            }
            
        # Remove internal details from the status before returning
        public_status = {k: v for k, v in status.items() if k not in ['start_time', 'last_update_time']}
        return public_status

    def process_repository_async(self, 
                                repo_url: str, 
                                branch: str = "main", 
                                force_refresh: bool = False,
                                extraction_params: Dict[str, Any] = None,
                                context_params: Dict[str, Any] = None,
                                on_complete: Optional[Callable[[str, Any], None]] = None) -> Dict[str, Any]:
        """
        Start asynchronous processing of a GitHub repository.
        
        Args:
            repo_url: URL of the GitHub repository
            branch: Branch name to process
            force_refresh: If True, reprocess even if already processed
            extraction_params: Parameters for the repository extractor
            context_params: Parameters for the LLM context updater
            on_complete: Optional callback for when processing completes
            
        Returns:
            A status dictionary with the immediate status information
        """
        repo_id = self._generate_repo_id(repo_url, branch)
        
        with self._lock:
            # Check if there's already a status for this repo
            current_status = self.status_tracker.get_status(repo_id)
            
            # If already complete and not forcing refresh, return the cached result
            if (current_status and 
                current_status.get("status") == ProcessingStatus.COMPLETE.value and 
                not force_refresh):
                self.logger.info(f"Using cached results for repository: {repo_url}")
                return current_status
                
            # If already processing and not forcing refresh, return current status
            if (current_status and 
                current_status.get("status") == ProcessingStatus.PROCESSING.value and
                not force_refresh):
                self.logger.info(f"Repository already processing: {repo_url}")
                return current_status
                
            # Reset the status if we're forcing a refresh
            if force_refresh and current_status:
                self.status_tracker.reset_repository(repo_id)
                
                # Cancel any active future for this repo
                if repo_id in self.active_futures:
                    future = self.active_futures[repo_id]
                    if future.running():
                        self.logger.warning(f"Canceling active processing for repository: {repo_url}")
                        future.cancel()
                    self.active_futures.pop(repo_id)
            
            # Initialize status for this repository
            status = self.status_tracker.start_processing(repo_id)
            
            # Start background processing
            self.logger.info(f"Starting background processing for repository: {repo_url}")
            extraction_params = extraction_params or {}
            context_params = context_params or {}
            
            # Submit the processing task to the thread pool
            future = self.thread_pool.submit(
                self._process_repository_task,
                repo_id=repo_id,
                repo_url=repo_url,
                branch=branch,
                extraction_params=extraction_params,
                context_params=context_params
            )
            
            # Add callback for when processing completes
            def _internal_callback(future):
                try:
                    result = future.result()
                    if on_complete:
                        on_complete(repo_id, result)
                except Exception as e:
                    self.status_tracker.mark_error(repo_id, str(e))
                finally:
                    with self._lock:
                        if repo_id in self.active_futures:
                            self.active_futures.pop(repo_id)
            
            future.add_done_callback(_internal_callback)
            self.active_futures[repo_id] = future
            
            return status
            
    def _process_repository_task(self, 
                                repo_id: str, 
                                repo_url: str, 
                                branch: str,
                                extraction_params: Dict[str, Any],
                                context_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a repository in a background thread.
        This is the main worker function that runs asynchronously.
        
        Args:
            repo_id: Unique identifier for the repository
            repo_url: URL of the GitHub repository
            branch: Branch name to process
            extraction_params: Parameters for the repository extractor
            context_params: Parameters for the LLM context updater
            
        Returns:
            The processed repository data with context
        """
        try:
            self.logger.info(f"Starting repository processing: {repo_url}")
            start_time = time.time()
            
            # Create repository extractor
            extractor = self.repository_extractor_class(
                repo_url=repo_url,
                **extraction_params
            )
            
            # Extract data from the repository
            self.logger.info(f"Extracting code from repository: {repo_url}")
            extracted_data = extractor.run()
            
            # Update progress after extraction
            self.status_tracker.update_progress(
                repo_id=repo_id, 
                files_processed=0,
                total_files=len(extracted_data)
            )
            
            # Create context updater
            context_updater = self.llm_context_updater_class(**context_params)
            
            # Process each file to generate context
            files_processed = 0
            for batch_start in range(0, len(extracted_data), 10):
                # Process in small batches
                batch_end = min(batch_start + 10, len(extracted_data))
                batch = extracted_data[batch_start:batch_end]
                
                # Update batch with context
                context_updater.update(batch)
                
                # Update progress
                files_processed += len(batch)
                self.status_tracker.update_progress(
                    repo_id=repo_id,
                    files_processed=files_processed,
                    total_files=len(extracted_data)
                )
                
                # Small delay to prevent excessive resources usage
                time.sleep(0.1)
            
            # Mark processing as complete
            processing_time = time.time() - start_time
            self.logger.info(f"Completed repository processing in {processing_time:.2f}s: {repo_url}")
            self.status_tracker.complete_processing(repo_id, extracted_data)
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error processing repository {repo_url}: {str(e)}")
            self.status_tracker.mark_error(repo_id, str(e))
            raise
    
    def wait_for_completion(self, repo_url: str, branch: str = "main", timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for a repository processing task to complete.
        
        This is a blocking operation and should not be used in API request handlers.
        It's primarily for testing or CLI applications.
        
        Args:
            repo_url: URL of the GitHub repository
            branch: Branch name (default: "main")
            timeout: Maximum time to wait in seconds (None for no timeout)
            
        Returns:
            Final status dictionary
        """
        repo_id = self._generate_repo_id(repo_url, branch)
        start_time = time.time()
        
        while True:
            status = self.status_tracker.get_status(repo_id)
            if not status:
                return {
                    "status": ProcessingStatus.NOT_STARTED.value,
                    "message": "Repository processing not started"
                }
                
            if status["status"] in [ProcessingStatus.COMPLETE.value, ProcessingStatus.ERROR.value]:
                return status
                
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return {
                    "status": "timeout",
                    "message": f"Timed out after waiting {timeout} seconds",
                    "current_status": status
                }
                
            time.sleep(1)  # Wait before checking again
            
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the processor and its thread pool.
        
        Args:
            wait: If True, wait for all tasks to complete before shutting down
        """
        self.logger.info(f"Shutting down AsyncRepositoryProcessor (wait={wait})")
        self.thread_pool.shutdown(wait=wait)