from utils.logging import configure_logging
from utils.loading import load_model_and_tokenizer
from utils.inference import generate_response
from utils.logging import logger


def main():
    """Main function to demonstrate model inference."""

    model_path = "data/Qwen2.5-3B-Instruct"

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_path)
    logger.info("Model loaded successfully!")

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    logger.info("Generating response...")
    response = generate_response(model, tokenizer, messages)
    logger.info("Generation completed!")
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
