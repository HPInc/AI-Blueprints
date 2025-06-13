"""
Repository Status Tracker

This module provides a thread-safe tracking mechanism for repository processing status.
It allows the API to immediately respond with the current status of repository processing
without blocking or timing out, even for large repositories.
"""

import threading
import time
from typing import Dict, Any, Optional
import logging
from enum import Enum

class ProcessingStatus(Enum):
    """Enumeration of possible repository processing statuses."""
    NOT_STARTED = "not_started"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"

class RepositoryStatusTracker:
    """
    Thread-safe tracker for repository processing status.
    
    This class provides methods to:
    - Start tracking a new repository processing job
    - Update the status and progress of a job
    - Retrieve the current status of a job
    - Store and retrieve the results of completed jobs
    
    All methods are thread-safe, using a lock to prevent race conditions.
    """
    
    def __init__(self):
        """Initialize the tracker with an empty status dictionary and a lock."""
        self.status_dict: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_status(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a repository.
        
        Args:
            repo_id: Unique identifier for the repository
            
        Returns:
            Dict with status information or None if not found
        """
        with self.lock:
            return self.status_dict.get(repo_id, None)
    
    def start_processing(self, repo_id: str) -> Dict[str, Any]:
        """
        Mark a repository as processing and initialize its status entry.
        
        Args:
            repo_id: Unique identifier for the repository
            
        Returns:
            The newly created status dictionary
        """
        with self.lock:
            status = {
                "status": ProcessingStatus.PROCESSING.value,
                "progress": 0,
                "files_processed": 0,
                "total_files": 0,
                "start_time": time.time(),
                "last_update_time": time.time()
            }
            self.status_dict[repo_id] = status
            self.logger.info(f"Started processing repository: {repo_id}")
            return status
    
    def update_progress(self, repo_id: str, files_processed: int, total_files: int) -> Dict[str, Any]:
        """
        Update the progress of a repository processing job.
        
        Args:
            repo_id: Unique identifier for the repository
            files_processed: Number of files processed so far
            total_files: Total number of files to process
            
        Returns:
            The updated status dictionary
        """
        with self.lock:
            if repo_id not in self.status_dict:
                self.start_processing(repo_id)
                
            status = self.status_dict[repo_id]
            status["files_processed"] = files_processed
            status["total_files"] = total_files
            
            # Calculate progress percentage
            if total_files > 0:
                status["progress"] = int((files_processed / total_files) * 100)
            else:
                status["progress"] = 0
                
            status["last_update_time"] = time.time()
            self.logger.debug(f"Updated progress for {repo_id}: {status['progress']}% ({files_processed}/{total_files})")
            return status
    
    def complete_processing(self, repo_id: str, context: Any) -> Dict[str, Any]:
        """
        Mark a repository as completely processed and store its context.
        
        Args:
            repo_id: Unique identifier for the repository
            context: The generated context for the repository
            
        Returns:
            The updated status dictionary
        """
        with self.lock:
            if repo_id not in self.status_dict:
                self.start_processing(repo_id)
                
            status = self.status_dict[repo_id]
            status["status"] = ProcessingStatus.COMPLETE.value
            status["progress"] = 100
            status["context"] = context
            status["completion_time"] = time.time()
            status["processing_duration"] = status["completion_time"] - status["start_time"]
            
            self.logger.info(f"Completed processing repository {repo_id} in {status['processing_duration']:.2f}s")
            return status
    
    def mark_error(self, repo_id: str, error_message: str) -> Dict[str, Any]:
        """
        Mark a repository processing job as failed with an error message.
        
        Args:
            repo_id: Unique identifier for the repository
            error_message: Description of the error that occurred
            
        Returns:
            The updated status dictionary
        """
        with self.lock:
            if repo_id not in self.status_dict:
                self.start_processing(repo_id)
                
            status = self.status_dict[repo_id]
            status["status"] = ProcessingStatus.ERROR.value
            status["error_message"] = error_message
            status["error_time"] = time.time()
            
            self.logger.error(f"Error processing repository {repo_id}: {error_message}")
            return status
            
    def reset_repository(self, repo_id: str) -> None:
        """
        Reset the status for a repository, removing it from the tracker.
        
        Args:
            repo_id: Unique identifier for the repository to reset
        """
        with self.lock:
            if repo_id in self.status_dict:
                self.status_dict.pop(repo_id)
                self.logger.info(f"Reset repository status for {repo_id}")
