import torch
import torch.nn as nn
from typing import List, Optional

from .encoders import AudioEncoder
from .encoders import TextEncoder
from .template import DATASET_TEMPLATE


class PromptLearner(nn.Module):
    def __init__(self, args, text_encoder, pengi):
        super().__init__()
        self.args = args
        self.pengi = pengi
        self.text_encoder = text_encoder
        self.device = args.device
        self.process_text = pengi.preprocess_text

        self.dataset_name = args.dataset_root.split('/')[1]
        print(f"Using dataset name: {self.dataset_name}")

        # safer default if template missing
        self.template = DATASET_TEMPLATE.get(self.dataset_name, "This is a sound of {}.")
        print(f"Using template: {self.template}")

        initial_features = self._get_init_features(args.classnames)  
        U, S, Vh = torch.linalg.svd(initial_features, full_matrices=False)  

        self.register_buffer("Vh_init", Vh.clone())       
        US = U @ torch.diag(S)                       
        self.register_buffer("US", US)

        # 3) Anchor cache
        orig_features = (US @ Vh)
        orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
        self.register_buffer("fixed_orig_features", orig_features) 

        self.ctx = nn.Parameter(Vh.clone()) 

    @torch.no_grad()
    def _get_init_features(self, classnames: List[str]) -> torch.Tensor:
        prompts = [self.template.format(name) for name in classnames]

        tokenized = self.process_text(prompts, enc_tok=True, add_text=False)
        input_ids = tokenized["input_ids"].to(self.device)
        attn_mask = tokenized["attention_mask"].to(self.device)
        with torch.no_grad():
            token_embeds = self.text_encoder.base.embeddings.token_embedding(input_ids)
            text = {"input_ids": input_ids, "inputs_embeds": token_embeds, "attention_mask": attn_mask}

            feats = self.text_encoder(text)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach()

    def forward(self):

        updated_features = self.US @ self.ctx
        updated_features = updated_features / updated_features.norm(dim=-1, keepdim=True)

        final_features = self.fixed_orig_features + updated_features
        final_features = final_features / final_features.norm(dim=-1, keepdim=True)

        return None, final_features, None

    @torch.no_grad()
    def get_prompts_for_new_classes(self, new_classnames: List[str]):

        if not hasattr(self, "_new_init_features_cache"):
            self._new_init_features_cache = {}

        cache_key = tuple(new_classnames)

        if cache_key in self._new_init_features_cache:
            new_init_features = self._new_init_features_cache[cache_key]
        else:
            new_init_features = self._get_init_features(new_classnames)
            new_init_features = new_init_features.to(self.ctx.device)
            self._new_init_features_cache[cache_key] = new_init_features

        rotation_matrix = self.Vh_init.t() @ self.ctx 

        V0 = self.Vh_init.t()                           
        proj = new_init_features @ V0                   
        beta = torch.norm(proj, dim=-1, keepdim=True) 

        new_tuned_raw = new_init_features @ rotation_matrix
        new_tuned_raw = new_tuned_raw / (new_tuned_raw.norm(dim=-1, keepdim=True))
        
        beta = beta.clamp(0.0, 1.0)

        final_new_features = new_init_features + beta * new_tuned_raw
        final_new_features = final_new_features / (final_new_features.norm(dim=-1, keepdim=True))
        return None, final_new_features, None


class CustomPENGI(nn.Module):
    def __init__(self, args, pengi):
        super().__init__()
        self.args = args
        self.device = args.device

        pengi_args = pengi.args
        self.audio_encoder = AudioEncoder(
            pengi_args.audioenc_name, pengi_args.out_emb, pengi_args.d_proj,
            pengi_args.sampling_rate, pengi_args.window_size, pengi_args.hop_size,
            pengi_args.mel_bins, pengi_args.fmin, pengi_args.fmax, pengi_args.classes_num,
            pengi_args.specaug, pengi_args.mixup,
            pengi_args.use_pretrained_audioencoder, pengi_args.freeze_audio_encoder_weights,
            pengi_args.use_precomputed_melspec, pengi_args.pretrained_audioencoder_path
        )

        self.text_encoder = TextEncoder(
            pengi_args.d_proj,
            pengi_args.text_model, pengi_args.transformer_embed_dim,
            pengi_args.freeze_text_encoder_weights
        )

        self.audio_encoder.load_state_dict(pengi.model.audio_encoder.state_dict())
        self.text_encoder.load_state_dict(pengi.model.caption_encoder.state_dict())

        self.audio_encoder.to(self.device).eval()
        self.text_encoder.to(self.device).eval()

        self.prompt_learner = PromptLearner(args, self.text_encoder, pengi)

    def forward(self, audio, is_test_b2n=False):
        audio_features = self.audio_encoder(audio)[0]
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        if is_test_b2n or self.args.cross_dataset:
            _, text_features, _ = self.prompt_learner.get_prompts_for_new_classes(self.args.all_classnames)
        else:
            _, text_features, _ = self.prompt_learner()

        logit_scale = 100.0
        logits = logit_scale * (audio_features @ text_features.t())
        return logits
