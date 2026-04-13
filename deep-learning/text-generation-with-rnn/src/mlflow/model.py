"""
Standalone Model class.

Business Logic Layer
- Handles RNN-based text generation with character-level modeling
- Manages model initialization, character encoding/decoding, and prediction logic
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import logging
from pyexpat import model
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import traceback

# Set up logger
logger = logging.getLogger(__name__)


class CharModel(nn.Module):
    def __init__(
        self,
        decoder,
        encoder,
        all_chars,
        num_hidden=256,
        num_layers=4,
        drop_prob=0.5,
        use_gpu=False,
        device="cpu",
    ):
        """Initializes CharModel

        Args:
            decoder: Assigns a unique integer to each character in a dictionary format
            encoder : Reverses the decoder dictionary, providing a mapping from characters to their respective assigned integers.
            all_chars: Set of unique characters found in the text.
            num_hidden: Number of hidden layers. Defaults to 256.
            num_layers: Number of layers. Defaults to 4.
            drop_prob: Regularization technique to prevent overfitting. Defaults to 0.5.
            use_gpu: If the model uses GPU. Defaults to False.
        """
        try:
            super().__init__()
            self.drop_prob = drop_prob
            self.num_layers = num_layers
            self.num_hidden = num_hidden
            self.use_gpu = use_gpu
            self.device = device

            self.all_chars = all_chars
            self.decoder = torch.load(decoder)
            self.encoder = torch.load(encoder)

            self.lstm = nn.LSTM(
                len(self.all_chars),
                num_hidden,
                num_layers,
                dropout=drop_prob,
                batch_first=True,
            )
            self.dropout = nn.Dropout(drop_prob)
            self.fc_linear = nn.Linear(num_hidden, len(self.all_chars))
            logger.info("CharModel initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing CharModel: {str(e)}")
            raise

    def forward(self, x, hidden):
        """Implementation of the CharModel logic, in which, the input passes through every step of the arquiteture

        Args:
            x: Input tensor with shape (batch size and senquency length) containing character indices.
            hidden: Tuple containing the inicial hidden states of the CharModel each with shape (batch size and senquency length).

        Returns:
            final_out: Output tensor representing the predicted logits for each character in the sequence.
            hidden: Tuple containing the final hidden states of the CharModel.
        """
        try:
            lstm_output, hidden = self.lstm(x, hidden)
            drop_output = self.dropout(lstm_output)
            drop_output = drop_output.contiguous().view(-1, self.num_hidden)
            final_out = self.fc_linear(drop_output)

            return final_out, hidden

        except Exception as e:
            logger.error(f"Error implementing CharModel logic: {str(e)}")

    def hidden_state(self, batch_size):
        """
        Initializes and returns the initial hidden state for a recurrent neural network (e.g., LSTM).

        This method creates zero-filled tensors for the hidden state (h_0) and cell state (c_0),
        supporting GPU execution if `self.use_gpu` is set to True.

        Args:
            batch_size: The number of sequences in the input batch, used to determine the tensor dimensions.

        Returns:
            Tuple: A tuple containing the hidden state and cell state tensors
            with shape (num_layers, batch_size, num_hidden). Returns None if an exception occurs, and logs the error.
        """
        try:
            if self.use_gpu:
                hidden = (
                    torch.zeros(self.num_layers, batch_size, self.num_hidden).to(
                        self.device
                    ),
                    torch.zeros(self.num_layers, batch_size, self.num_hidden).to(
                        self.device
                    ),
                )
            else:
                hidden = (
                    torch.zeros(self.num_layers, batch_size, self.num_hidden),
                    torch.zeros(self.num_layers, batch_size, self.num_hidden),
                )

            return hidden
        except Exception as e:
            logger.error(
                f"Error Initializing and returning the initial hidden state: {str(e)}"
            )


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles RNN-based text generation with character-level modeling.
    """

    def __init__(
        self,
        config: dict,
        model_state_dict_path: str,
        decoder_path: str,
        encoder_path: str,
        all_chars: set,
        device: str,
    ):
        """
        Direct dependency injection - no MLflow context.
        Extract all initialization logic from original RNNModel.

        Args:
            config: Configuration dictionary
            model_state_dict_path: Path to the trained model state dictionary
            decoder_path: Path to the character decoder dictionary
            encoder_path: Path to the character encoder dictionary
            all_chars: Set of unique characters found in the training text
            device: Device to run the model on ("cuda" or "cpu")
        """
        try:
            self.config = config
            self.all_chars = all_chars
            self.device = device

            # Initialize the CharModel with architecture parameters
            self.model = CharModel(
                all_chars=all_chars,
                num_hidden=512,
                num_layers=3,
                drop_prob=0.5,
                use_gpu=True if device == "cuda" else False,
                decoder=decoder_path,
                encoder=encoder_path,
                device=device,
            )

            # Load the trained model state dictionary
            self.model.load_state_dict(torch.load(model_state_dict_path))
            self.model.eval()

            logger.info("Model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Model: {str(e)}")
            raise

    def one_hot_encoder(self, encoded_text, num_uni_chars):
        """
        Convert categorical data into a fixed-size vector of numerical values.

        Args:
            encoded_text: Batch of encoded text.
            num_uni_chars: Number of unique characters

        Returns:
            One-hot encoded representation of the input text
        """
        try:
            one_hot = np.zeros((encoded_text.size, num_uni_chars))
            one_hot = one_hot.astype(np.float32)
            one_hot[np.arange(one_hot.shape[0]), encoded_text.flatten()] = 1.0
            one_hot = one_hot.reshape((*encoded_text.shape, num_uni_chars))

            return one_hot

        except Exception as e:
            logger.error(f"Error converting categorical data: {str(e)}")
            raise

    def predict_next_char(self, char, hidden=None, k=3):
        """
        Predicts the next character given an input character and the current hidden state.

        This method encodes the input character, feeds it through the trained character-level
        language model (e.g., LSTM), and samples from the top-k most probable characters
        to determine the next one. It also returns the updated hidden state for sequential prediction.

        Args:
            char: The input character to start prediction from.
            hidden: Current hidden state of the model. Each tensor has shape (num_layers, batch_size, num_hidden).
                If None, a new hidden state should be initialized before calling this method.
            k: Number of top predictions to sample from.

        Returns:
            A tuple containing the predicted next character and the updated hidden state.
        """
        try:
            encoded_text = self.model.encoder[char]
            encoded_text = np.array([[encoded_text]])
            encoded_text = self.one_hot_encoder(encoded_text, len(self.model.all_chars))
            inputs = torch.from_numpy(encoded_text)

            if self.model.use_gpu:
                inputs = inputs.to(self.device)

            hidden = tuple([state.data for state in hidden])
            lstm_out, hidden = self.model(inputs, hidden)
            probs = F.softmax(lstm_out, dim=1).data

            if self.model.use_gpu:
                probs = probs.cpu()

            probs, index_positions = probs.topk(k)
            index_positions = index_positions.numpy().squeeze()
            probs = probs.numpy().flatten()
            probs = probs / probs.sum()
            char = np.random.choice(index_positions, p=probs)

            return self.model.decoder[char], hidden
        except KeyError as e:
            logger.error(f"Character not found in encoder: {str(e)}")
            logger.error(
                f"Available characters in encoder: {list(self.model.encoder.keys())[:20]}..."
            )  # Show first 20
            raise KeyError(f"Character '{char}' not found in encoder dictionary") from e
        except Exception as e:
            logger.error(f"Error predicting next char: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def generate_text(self, seed, size, k=3):
        """
        Generates a sequence of text using the trained character-level language model.

        Starting from a seed string, this method uses the model to predict the next character
        one at a time, feeding each predicted character back into the model. It continues
        this process until the desired output length is reached.

        Args:
            seed: The initial sequence of characters used to start the text generation.
            size: The number of characters to generate after the seed.
            k: Number of top character predictions to consider for sampling at each step.

        Returns:
            The full generated text including the seed and the newly predicted characters.
        """
        try:
            if self.model.use_gpu:
                self.model.to(self.device)
            else:
                self.model.cpu()

            self.model.eval()
            output_chars = [c for c in seed]
            hidden = self.model.hidden_state(1)

            for char in seed:
                char, hidden = self.predict_next_char(char, hidden, k=k)

            output_chars.append(char)
            for i in range(size):
                char, hidden = self.predict_next_char(output_chars[-1], hidden, k=k)
                output_chars.append(char)

            return "".join(output_chars)

        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def predict(self, model_input, params=None):
        """
        Core business logic extracted from original RNNModel predict method.
        Remove context parameter - use instance variables instead.
        Must return same pandas.DataFrame structure as original.

        Args:
            model_input: Input data dictionary containing 'initial_word' and 'size' keys
            params: Optional parameters (not used in this implementation)

        Returns:
            pandas.DataFrame containing the generated text result
        """
        try:
            # Extract inputs from model_input dictionary

            initial_word = model_input["initial_word"][0]
            size = model_input["size"][0]
            output = self.generate_text(seed=initial_word, size=size)

            return pd.DataFrame({"generated_text": [output]})

        except Exception as e:
            error_details = f"Predict method error: {str(e)}\nFull traceback:\n{traceback.format_exc()}"
            logger.error(error_details)
            # Return detailed error in DataFrame format for debugging
            error_df = pd.DataFrame(
                {
                    "generated_text": [
                        f"Error generating text: {str(e)} (Check logs for details)"
                    ]
                }
            )
            return error_df
