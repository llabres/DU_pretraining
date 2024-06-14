"""
Original: https://github.com/rubenpt91/MP-DocVQA-Framework/blob/master/models/_modules.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Config
from transformers import AutoModel
from torch.nn import CrossEntropyLoss
from torch.nn import LayerNorm as BertLayerNorm

def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):
    # Source: https://pub.aimind.so/creating-sinusoidal-positional-embedding-from-scratch-in-pytorch-98c49e153d6
    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim #d_model=head_num*d_k, not d_q, d_k, d_v

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings


class CustomT5Config(T5Config):
    def __init__(self, max_2d_position_embeddings=1024,  **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.hidden_dropout_prob = 0.1
        self.layer_norm_eps = 1e-12


class SpatialEmbeddings(nn.Module):
    """
    Spatial embedding by summing x, y, w, h projected by nn.Embedding to hidden size.
    """

    def __init__(self, config):
        super(SpatialEmbeddings, self).__init__()

        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model
        )

        self.LayerNorm = BertLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        #self.spatial_emb_matcher = MLP(config.d_model, 0, config.d_model, 1)

        self.config = config

    def forward(self, bbox):

        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

        # h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])  # TODO Remove width and height to test how much important are they.
        # w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])  # TODO Remove width and height to test how much important are they.

        embeddings = (
                left_position_embeddings
                + upper_position_embeddings
                + right_position_embeddings
                + lower_position_embeddings
                # + h_position_embeddings
                # + w_position_embeddings
        )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #embeddings = self.spatial_emb_matcher(embeddings)
        return embeddings
    
class ContinuousSpatialEmbeddings(nn.Module):
    def __init__(self, config):
        super(ContinuousSpatialEmbeddings, self).__init__()

        self.x_position_embeddings = nn.Linear(1, config.d_model, bias=False)
        self.y_position_embeddings = nn.Linear(1, config.d_model, bias=False)


        self.LayerNorm = BertLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        #self.spatial_emb_matcher = MLP(config.d_model, 0, config.d_model, 1)

        self.config = config

    def forward(self, bbox):
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0].unsqueeze(-1))
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1].unsqueeze(-1))
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2].unsqueeze(-1))
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3].unsqueeze(-1))

        embeddings = (
                left_position_embeddings
                + upper_position_embeddings
                + right_position_embeddings
                + lower_position_embeddings
        )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #embeddings = self.spatial_emb_matcher(embeddings)
        return embeddings



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class VisualEmbeddings(nn.Module):
    r"""
    Construct the embeddings from patch. In `Pix2Struct` the input is different from classic Vision-transformer models.
    Here the input is a sequence of `seq_len` flattened patches that also combines padding patches (tokens). Each patch
    is represented by a vector of `hidden_size` values.
    """

    def __init__(self, config) -> None:
        super().__init__()
        
        self.patch_projection = nn.Linear(config.patch_embed_hidden_size, config.d_model)

        self.dropout = nn.Dropout(config.dropout_rate)

        self.continuous_spatial_embeddings = config.continuous_spatial_embeddings
    
    def get_visual_boxes(self, row_indices, col_indices, image_width, image_height):
        width_patches = torch.max(image_width, dim=-1, keepdim=True).values
        height_patches = torch.max(image_height, dim=-1, keepdim=True).values 

        width_patches[width_patches == 0] = 1
        height_patches[height_patches == 0] = 1

        boxes = torch.stack([
            col_indices / width_patches,
            row_indices / height_patches,
            (col_indices + 1) / width_patches,
            (row_indices + 1) / height_patches,
        ], dim=-1)

        boxes[boxes<0] = 0

        assert torch.all(boxes >= 0) and torch.all(boxes <= 1)#f"Boxes should be in the range [0, 1] but got {boxes.min()} and {boxes.max()}"

        if not self.continuous_spatial_embeddings:
            boxes = boxes * 1000
            boxes = boxes.to(torch.long)
        
        return boxes

    def forward(self, flattened_patches: torch.Tensor) -> torch.Tensor:
        # the row and column indices are stored in the first and second position of the flattened_patches
        # flattened_patches: `batch_size`, `seq_len`, `hidden_size` + 2
        image_width, image_height = flattened_patches[:, :, 0], flattened_patches[:, :, 1]
        boxes = self.get_visual_boxes(flattened_patches[:, :, 2] - 1, flattened_patches[:, :, 3] - 1, image_width, image_height)
        flattened_patches = flattened_patches[:, :, 4:]

        embeddings = self.patch_projection(flattened_patches)

        embeddings = self.dropout(embeddings)

        return embeddings, boxes
        

class OldVisualEmbeddings(nn.Module):

    def __init__(self, config, finetune=False):
        super(OldVisualEmbeddings, self).__init__()
        self.image_size = (config.image_size, config.image_size) if type(config.image_size) == int else config.image_size
        self.image_model = AutoModel.from_pretrained(config.visual_module_config['model_weights'])
        self.visual_emb_matcher = MLP(self.image_model.config.d_model, 0, self.image_model.config.d_model, 1)
        # TODO: modify config file so patch size is in it
        #self.patch_size = config.visual_module_config['patch_size']
        self.patch_size = (16, 16)

        if not finetune:
            self.freeze()
        
        self.continuous_spatial_embeddings = config.continuous_spatial_embeddings
        self.scale_boxes = 1000 if self.continuous_spatial_embeddings else 1

    def freeze(self):
        for p in self.image_model.parameters():
            p.requires_grad = False

    def get_visual_boxes(self, num_pages=1):
        boxes = torch.tensor([[x / self.patch_size[0], y / self.patch_size[1], (x + 1) / self.patch_size[0], (y + 1) / self.patch_size[1]] for y in range(0, self.image_size[1]//self.patch_size[1]) for x in range(0, self.image_size[0]//self.patch_size[0])], dtype=torch.float32)
        boxes = torch.cat([boxes, torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)], dim=0) # Adding box for CLS token
        boxes = boxes.unsqueeze(dim=0).expand([num_pages, -1, -1])
        boxes = boxes * self.scale_boxes
        boxes = boxes.to(torch.long) if not self.continuous_spatial_embeddings else boxes
        boxes = boxes.to(self.image_model.device)
        return boxes

    def forward(self, images, page_idx_mask=None):
        vis_embeddings = self.image_model(images)
        vis_embeddings = vis_embeddings.last_hidden_state  # BS; 14x14+CLS (197); 768 (hidden size)
        vis_embeddings = self.visual_emb_matcher(vis_embeddings)

        if page_idx_mask is not None:
            vis_attention_mask = torch.zeros(vis_embeddings.shape[:2], dtype=torch.long).to(self.image_model.device)
            vis_attention_mask[page_idx_mask] = 1
        else:
            vis_attention_mask = torch.ones(vis_embeddings.shape[:2], dtype=torch.long).to(self.image_model.device)

        return vis_embeddings, vis_attention_mask


class RetrievalModule(nn.Module):

    def __init__(self, config):
        super(RetrievalModule, self).__init__()

        self.page_retrieval = nn.Linear(config.max_doc_pages * config.page_tokens * config.hidden_size, config.max_doc_pages)
        # TODO Check if BinaryCrossEntropy allows to extend to longer sequences.

        if config.page_retrieval_config['loss'].lower() in ['ce', 'crossentropy', 'crossentropyloss']:
            self.retrieval_criterion = CrossEntropyLoss()

        self.retrieval_loss_weight = config.page_retrieval_config['loss_weight']

    def forward(self, document_embeddings, answer_page_idx):
        document_embeddings = document_embeddings.view([len(document_embeddings), -1])
        # document_embeddings = F.pad(document_embeddings, (0, self.page_retrieval.in_features-document_embeddings.shape[-1]), "constant", 0)  # In case is the last batch

        try:
            ret_logits = self.page_retrieval(document_embeddings)  # 10*2*512

        except:
            pad_document_embeddings = torch.zeros([len(document_embeddings), self.page_retrieval.in_features], dtype=document_embeddings.dtype, device=document_embeddings.device)
            pad_document_embeddings[:, :document_embeddings.shape[-1]] = document_embeddings
            ret_logits = self.page_retrieval(pad_document_embeddings.to())  # 10*2*512

        ret_loss = self.retrieval_criterion(ret_logits, answer_page_idx) * self.retrieval_loss_weight if answer_page_idx is not None else None

        return ret_loss, ret_logits