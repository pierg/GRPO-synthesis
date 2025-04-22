import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from jinja2 import Environment
from tokenizers import Encoding
from tokenizers import Tokenizer as TokenizerBase


class Tokenizer:
    """Tokenizer class that supports chat templates using the jinja2 templating engine.
    
    This class extends the base tokenizer functionality to handle chat-style conversations
    with proper formatting and special tokens. It's designed to work with language models
    that expect structured chat inputs.
    """

    def __init__(self, tokenizer_path: str):
        """Initialize the tokenizer with configuration and templates.
        
        Args:
            tokenizer_path: Path to the tokenizer.json file
        """
        # Load the tokenizer configuration file
        tokenizer_config_path = Path(tokenizer_path).parent / "tokenizer_config.json"
        self.tokenizer_config = json.load(open(tokenizer_config_path))
        
        # Initialize the base tokenizer from the provided path
        self.tokenizer = TokenizerBase.from_file(tokenizer_path)
        
        # Set up the chat template using jinja2
        self.chat_template = Environment().from_string(
            self.tokenizer_config["chat_template"]
        )
        
        # Extract special tokens and their IDs from the configuration
        self.eos_token = self.tokenizer_config["eos_token"]  # End of sequence token
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        self.pad_token = self.tokenizer_config["pad_token"]  # Padding token
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)

    def encode_chat(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Encodes a list of chat messages into a formatted string.
        
        Uses the chat template to format the messages according to the model's
        expected input format. The template typically adds role markers and
        formatting for system, user, and assistant messages.
        
        Args:
            messages: List of chat messages, each with 'role' and 'content' keys
            add_generation_prompt: Whether to add the assistant's response prompt
        
        Returns:
            Formatted chat string
        """
        return self.chat_template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt
        )

    def encode_chat_with_response_prompt(
        self, messages: List[Dict[str, str]], prompt: str
    ) -> str:
        """Encodes chat messages and appends a response prompt.
        
        This is used when we want the model to start generating a response
        after seeing the chat history. The prompt typically includes the
        beginning of the model's response.
        
        Args:
            messages: List of chat messages, each with 'role' and 'content' keys
            prompt: Response prompt to append after the chat messages
        
        Returns:
            Formatted chat string with response prompt
        """
        return self.encode_chat(messages) + prompt

    def tokenize(self, text: str) -> Encoding:
        """Converts text into token IDs using the base tokenizer.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Tokenizer encoding containing token IDs and other information
        """
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Converts token IDs back into text.
        
        Args:
            token_ids: List of token IDs to convert to text
            skip_special_tokens: Whether to skip special tokens in the output
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self) -> int:
        """Returns the size of the tokenizer's vocabulary."""
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self) -> Dict[str, int]:
        """Returns a dictionary mapping special token names to their IDs."""
        return {
            "eos_token": self.eos_token_id,
            "pad_token": self.pad_token_id
        }

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Converts tokens to their corresponding IDs.
        
        Args:
            tokens: Single token string or list of tokens
            
        Returns:
            Single token ID or list of token IDs
        """
        if isinstance(tokens, str):
            return self.tokenizer.token_to_id(tokens)
        return [self.tokenizer.token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Converts token IDs to their corresponding tokens.
        
        Args:
            ids: Single token ID or list of token IDs
            
        Returns:
            Single token string or list of tokens
        """
        if isinstance(ids, int):
            return self.tokenizer.id_to_token(ids)
        return [self.tokenizer.id_to_token(id) for id in ids]

    def convert_token_to_readable(self, token: str) -> str:
        """Converts a token to its readable form by handling special characters.
        
        Args:
            token: Token string to convert
            
        Returns:
            Readable version of the token
        """
        if token.startswith("Ġ"):
            return " " + token[1:]
        elif token == "Ċ":
            return "\n"
        return token

    def convert_tokens_to_readable(self, tokens: Union[str, List[str]]) -> Union[str, List[str]]:
        """Converts tokens to their readable form by handling special characters.
        
        Args:
            tokens: Single token string or list of tokens
            
        Returns:
            Readable version of the token(s)
        """
        if isinstance(tokens, str):
            return self.convert_token_to_readable(tokens)
        return [self.convert_token_to_readable(token) for token in tokens] 