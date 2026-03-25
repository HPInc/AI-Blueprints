"""
QwenOmniAgent implementation for audio question answering.

This module provides a unified adapter for the Qwen2.5-Omni-7B model used across
both the notebook workflow and MLflow deployment.
"""

import re
import numpy as np
import torch
import soundfile as sf
from typing import List, Dict, Any


class QwenOmniAgent:
    """
    Minimal adapter for Qwen2.5-Omni-7B audio reasoning.

    This class wraps the Qwen Omni processor and model to provide
    interface for answering questions based on audio segments.

    Key features:
    - Processes multiple audio segments with CLAP-based retrieval
    - Uses task-specific system prompt for precise analysis
    - Generates coherent responses with proper token limits
    """

    def __init__(self, processor, model):
        """
        Initialize the agent with Qwen Omni processor and model.

        Args:
            processor: Qwen2_5OmniProcessor instance for text/audio processing
            model: Qwen2_5OmniThinkerForConditionalGeneration instance
        """
        self.processor = processor
        self.model = model
        self.device = getattr(
            model, "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

    def answer(
        self,
        question: str,
        audio_hits: List[Dict[str, Any]],
        return_audio: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate an answer to a question based on retrieved audio segments.

        Args:
            question: The question to answer
            audio_hits: List of dicts containing:
                - file_path: Path to the audio file
                - file_name: Name of the audio file
                - start_s: Start time in seconds
                - end_s: End time in seconds
                - score: Relevance score from retrieval
                - wav_path: Path to WAV file
            return_audio: If True, include generated audio in response (future feature)

        Returns:
            Dictionary containing:
                - answer: Generated text answer
                - evidence: List of evidence segments with timestamps and scores
        """
        # Build conversation with system prompt and user's question + audio clips
        user_content = [{"type": "text", "text": f"Question: {question}"}]

        # Attach each retrieved audio segment
        usable = 0
        for h in audio_hits:
            audio_full, sr = sf.read(h["wav_path"])
            if audio_full.ndim == 2:
                audio_full = audio_full.mean(axis=1)
            s0, s1 = int(h["start_s"] * sr), int(h["end_s"] * sr)
            s0 = max(0, min(s0, len(audio_full)))
            s1 = max(0, min(s1, len(audio_full)))
            if s1 <= s0:
                continue  # Skip empty windows
            seg = audio_full[s0:s1].astype(np.float32)
            user_content.append({"type": "audio", "audio": seg, "sampling_rate": sr})
            usable += 1

        # If no usable segments, return early
        if usable == 0:
            return {"answer": "Not found in audio.", "evidence": []}

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a precise analyst. Listen to the audio clips and answer using only their content. Keep timestamps where helpful.",
                    }
                ],
            },
            {"role": "user", "content": user_content},
        ]

        # 1) Create the text prompt from the chat template
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # 2) Collect multimodal blobs (audio) from the conversation
        from qwen_omni_utils import process_mm_info

        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        # 3) Pack tensors for the model - IMPORTANT: add dtype conversion
        inputs = (
            self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            .to(self.model.device)
            .to(self.model.dtype)
        )

        # 4) Generate with parameters
        token = getattr(self.processor, "tokenizer", None)
        eos_id = getattr(token, "eos_token_id", None)
        pad_id = getattr(token, "pad_token_id", eos_id)

        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                use_audio_in_video=False,
                max_new_tokens=768,  # Sufficient tokens for complete responses
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
                repetition_penalty=1.05,  # Light penalty to maintain coherence
            )

        # Decode only the new tokens (exclude prompt)
        seq = gen.sequences
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = seq[:, prompt_len:]

        answer = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        # Clean up any stray role headers
        answer = re.sub(
            r"^(?:\d+\s*)?(?:Human:|User:|Assistant:|System:)\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        ).strip()

        # Build evidence list
        evidence = [
            {
                "file_name": h["file_name"],
                "file_path": h["file_path"],
                "start_s": h["start_s"],
                "end_s": h["end_s"],
                "score": h.get("score_mmr", h.get("score", 0.0)),
            }
            for h in audio_hits
        ]

        return {
            "answer": answer if answer else "Not found in audio.",
            "evidence": evidence,
        }
