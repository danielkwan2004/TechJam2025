# llm.py
from transformers import AutoTokenizer, AutoModel
import torch

class LegalBERTEmbedder:
    """
    Local LegalBERT embedder: mean-pooled token embeddings + L2 normalization.
    Uses AutoModel (not AutoModelForPreTraining) because we only need embeddings.
    """
    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Device info (safe print even if no GPU)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                print("CUDA device:", torch.cuda.get_device_name(0))
            except Exception:
                print("CUDA device: available")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use base model for embeddings (not the pretraining heads)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def embed(self, texts, max_length: int = 512, normalize: bool = True):
        """
        Returns a CPU tensor of shape (N, hidden_size).
        """
        if isinstance(texts, str):
            texts = [texts]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)  # last_hidden_state
        last_hidden = outputs.last_hidden_state                    # (B, L, H)
        mask = batch["attention_mask"].unsqueeze(-1).type_as(last_hidden)  # (B, L, 1)

        # Mean pool
        summed = (last_hidden * mask).sum(dim=1)                   # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)                   # (B, 1)
        mean = summed / counts

        # L2 normalize
        if normalize:
            mean = torch.nn.functional.normalize(mean, p=2, dim=1)

        return mean.cpu()


if __name__ == "__main__":
    # Quick smoke test
    embedder = LegalBERTEmbedder()
    emb = embedder.embed(["Hello, my dog is cute"])
    print("Embedding shape:", emb.shape)  # e.g., torch.Size([1, 768])
