import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


# CLIP model with finetuned projection and emphasis on text
class VLM(nn.Module):
    def __init__(self, num_envs, device, vlm_feature_size=64, model_id="openai/clip-vit-base-patch16", text_alpha=2.0):
        super(VLM, self).__init__()
        self.num_envs = num_envs
        self.device = device
        self.text_alpha = text_alpha

        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False  # freeze CLIP

        hidden_size = self.model.config.projection_dim  # typically 512
        img_feature_size = 64
        text_feature_size = hidden_size

        # Projection MLPs for image and text embeddings
        self.img_proj = nn.Sequential(
            nn.Linear(hidden_size, img_feature_size),
            nn.LayerNorm(img_feature_size),
            nn.Tanh()
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(hidden_size, text_feature_size),
            nn.LayerNorm(text_feature_size),
            nn.Tanh()
        )

        # Final fusion
        self.fc = nn.Linear(img_feature_size + text_feature_size, vlm_feature_size)

    def forward(self, image, text_prompt):
        inputs = self.processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
            do_rescale=False,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_emb = outputs.image_embeds     # [B, 512]
            text_emb = outputs.text_embeds       # [B, 512]

        # Normalize
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Project and scale
        image_feat = self.img_proj(image_emb)
        text_feat = self.txt_proj(text_emb) * self.text_alpha  # Add weighting here

        fused = torch.cat([image_feat, text_feat], dim=-1)     # [B, 1024]
        final_feat = self.fc(fused)                            # [B, vlm_feature_size]

        return final_feat
