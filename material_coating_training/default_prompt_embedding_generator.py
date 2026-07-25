import torch
import os
from diffusers import FluxControlPipeline


def generate_and_save_empty_embeddings(text_encoding_pipeline, save_path="empty_embeddings.pt"):
    """
    Generate embeddings for empty prompt and save them to file
    """
    print("Generating empty prompt embeddings...")

    text_encoding_pipeline = text_encoding_pipeline.to("cuda")

    with torch.no_grad():
        # Encode empty prompt
        empty_prompt = [""]  # Single empty string
        prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
            empty_prompt, prompt_2=None, max_sequence_length=10
        )

    # Save to file
    embeddings_dict = {
        "prompt_embeds": prompt_embeds.cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
        "text_ids": text_ids.cpu(),
        "shapes": {
            "prompt_embeds": list(prompt_embeds.shape),
            "pooled_prompt_embeds": list(pooled_prompt_embeds.shape),
            "text_ids": list(text_ids.shape)
        }
    }

    torch.save(embeddings_dict, save_path)
    print(f"Empty embeddings saved to {save_path}")
    print(
        f"Shapes - prompt_embeds: {prompt_embeds.shape}, pooled_prompt_embeds: {pooled_prompt_embeds.shape}, text_ids: {text_ids.shape}")

    return embeddings_dict


def generate_word_embeddings(text_encoding_pipeline, words, save_path="word_embeddings.pt"):
    """
    Generate embeddings for specific words and save to single file

    Args:
        text_encoding_pipeline: The text encoding pipeline
        words: List of words to generate embeddings for
        save_path: Path to save the embeddings dictionary
    """
    text_encoding_pipeline = text_encoding_pipeline.to("cuda")
    all_embeddings = {}

    for word in words:
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                [word], prompt_2=None, max_sequence_length=1  # 10
            )

            # all_embeddings[word] = {
            #     "prompt_embeds": prompt_embeds.cpu(),
            #     "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
            #     "text_ids": text_ids.cpu()
            # }
            all_embeddings[word] = prompt_embeds.cpu().squeeze(0)[0]

    torch.save(all_embeddings, save_path)
    print(f"Word embeddings saved to {save_path}")
    return all_embeddings


def load_empty_embeddings(file_path="empty_embeddings.pt"):
    """
    Load pre-generated empty embeddings from file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Empty embeddings file not found: {file_path}")

    embeddings_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    # print(f"Shapes - prompt_embeds: {embeddings_dict['shapes']['prompt_embeds']}, "
    #       f"pooled_prompt_embeds: {embeddings_dict['shapes']['pooled_prompt_embeds']}, "
    #       f"text_ids: {embeddings_dict['shapes']['text_ids']}")

    return embeddings_dict


# Usage in training loop:
def get_empty_embeddings_for_batch(batch_size, empty_embeddings_dict, device):
    """
    Expand the single empty embedding to match batch size
    """
    # Get single embeddings and expand to batch size
    single_prompt_embeds = empty_embeddings_dict["prompt_embeds"].to(device)
    single_pooled_embeds = empty_embeddings_dict["pooled_prompt_embeds"].to(device)
    single_text_ids = empty_embeddings_dict["text_ids"].to(device)

    prompt_embeds = single_prompt_embeds.repeat(batch_size, 1, 1)
    pooled_prompt_embeds = single_pooled_embeds.repeat(batch_size, 1)
    text_ids = single_text_ids.repeat(batch_size, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoding_pipeline = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=None, vae=None, torch_dtype=torch.bfloat16
    ).to(device)

    generate_and_save_empty_embeddings(text_encoding_pipeline)

    words = ["metallic", "dielectric", "transmissive", "opaque", "thickness", "roughness",]
    generate_word_embeddings(text_encoding_pipeline, words, "material_default_embeddings.pt")
