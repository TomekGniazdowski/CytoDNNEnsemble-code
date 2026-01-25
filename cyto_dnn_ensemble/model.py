import torch
import torch.nn as nn

from cyto_dnn_ensemble.classification_heads import NNFinalHeadAttention, NNFinalHeadClf


class CytoDNNEnsembleModel(nn.Module):
    """Ensemble of two feature extractors and final classifier."""

    def __init__(
        self,
        cnn_1d_extractor: nn.Module,
        latent_space_extractor: nn.Module,
        final_classifier_type: str,
        final_model_clf_hidden_dim: int,
        train_backbone: bool
    ):

        super().__init__()
        
        self.cnn_1d_extractor = cnn_1d_extractor
        for param in self.cnn_1d_extractor.parameters():
            param.requires_grad = train_backbone
        
        self.latent_space_extractor = latent_space_extractor
        for param in self.latent_space_extractor.parameters():
            param.requires_grad = train_backbone
        
        self.final_classifier_type = final_classifier_type
        if final_classifier_type == 'final_model_clf':
            self._clf = NNFinalHeadClf(hidden_dim=final_model_clf_hidden_dim)
        elif final_classifier_type == 'final_model_attention':
            self._clf = NNFinalHeadAttention()
        else:
            raise ValueError("Incorrect final model name.")
        
        self._attention_maps_path = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """1D CNN extractor"""
        y_ce = self.cnn_1d_extractor(x)
        
        """Latent space extractor"""
        y_le = self.latent_space_extractor(x)

        y = torch.stack([y_ce, y_le], dim=1).double()
        
        """Final classifier"""
        if self.final_classifier_type == 'final_model_clf':
            y = self._clf(y)        
        elif self.final_classifier_type == 'final_model_attention':
            attention_coefficients = self._clf(y)
            if self._attention_maps_path is not None:
                torch.save(attention_coefficients, self._attention_maps_path)
            y = (y * attention_coefficients).sum(dim=1)
            y = torch.clamp(y, min=0., max=1.)
            
        return  y