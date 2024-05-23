import numpy as np
import torch
import torch.nn.functional as F
import copy

from torch import nn
from recbole.model.layers import TransformerEncoder
from recbole.model.abstract_recommender import SequentialRecommender
from IPython import embed
from entmax import entmax_bisect
from recbole.model.layers import FeedForward, MultiHeadAttention

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils_seq import reverse_packed_sequence

class TransNet(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()

        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['embedding_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.entmax_alpha = config['entmax_alpha'] if 'entmax_alpha' in config else -1
        self.use_highway = config['use_highway']
        self.max_repeat = config['max_repeat'] if 'max_repeat' in config else 2

        self.position_embedding = nn.Embedding(dataset.field2seqlen['item_id_list']+1, self.hidden_size) # 0 for mean pooling
        self.repeat_embedding = nn.Embedding(self.max_repeat + 1, self.hidden_size, padding_idx=0)
        self.trm_encoder = TransformerEncoder(
            config,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.highway_fn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.importance_fn = nn.Linear(self.hidden_size, 1, bias=False)
        self.entmax_with_mean = config['entmax_with_mean'] if 'entmax_with_mean' in config else False
        if self.entmax_with_mean:
            self.importance_fn = nn.Linear(self.hidden_size * 2, 1)

        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq, bidirectional=True):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding

        is_repeat = (item_seq.unsqueeze(1) == item_seq.unsqueeze(2)).sum(dim=-1) # [B, L]
        is_repeat = torch.where(mask, is_repeat, 0)
        is_repeat = torch.clamp(is_repeat, max=self.max_repeat)
        repeat_emb = self.repeat_embedding(is_repeat)
        input_emb = input_emb + repeat_emb

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        if self.use_highway:
            gate_vector = torch.sigmoid(self.highway_fn(torch.cat([item_emb, output], dim=-1))) # [B, L, D]
            output = gate_vector * item_emb + (1 - gate_vector) * output
        
        if self.entmax_with_mean:
            output_w_mean = torch.cat([output[:, 0, :].unsqueeze(1).repeat(1, output.size(1), 1), output], dim=-1)
            importance = self.importance_fn(output_w_mean).to(torch.double)
        else:
            importance = self.importance_fn(output).to(torch.double)
        importance = torch.where(mask.unsqueeze(-1), importance, -9e15)

        alpha_for_entmax = self.entmax_alpha
        gamma_prob = entmax_bisect(importance, alpha_for_entmax, dim=1)

        return output, gamma_prob

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class MIASREC(SequentialRecommender):
    def __init__(self, config, dataset):
        super(MIASREC, self).__init__(config, dataset)
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.device = config['device']
        self.loss_type = config['loss_type']

        self.sess_dropout = nn.Dropout(config['sess_dropout'])
        self.item_dropout = nn.Dropout(config['item_dropout'])
        self.temperature = config['temperature']

        # item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

        # parameters initialization
        self._reset_parameters()

        self.net = TransNet(config, dataset)
        self.num_interest = config['num_interest']
        self.beta_logit = config['beta_logit']
        
        self.use_mean = config['use_mean'] if 'use_mean' in config else False
        self.extractor = config['extractor']
        self.aggregator = config['aggregator']

        self.use_entmax = config['use_entmax'] if 'use_entmax' in config else False

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def ave_net(self, item_seq):
        mask = item_seq.gt(0)
        alpha = mask.to(torch.float) / mask.sum(dim=-1, keepdim=True)
        return alpha.unsqueeze(-1)

    def forward(self, item_seq):
        # reverse the item sequence without padding (padding is 0)
        with torch.no_grad():
            lengths = torch.sum(item_seq != 0, dim=1)
            new_item_seq = pack_padded_sequence(item_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
            new_item_seq = reverse_packed_sequence(new_item_seq)
            new_item_seq, _ = pad_packed_sequence(new_item_seq, batch_first=True, padding_value=0)
            item_seq = new_item_seq

        x = self.item_embedding(item_seq) # [B, L, D]
        
        if self.use_mean:
            alpha = self.ave_net(item_seq)
            mean_emb = torch.sum(alpha * x, dim=1) # [B, D]
            
            x = torch.cat([mean_emb.unsqueeze(1), x], dim=1) # [B, L+1, D]
            x = self.sess_dropout(x) # [B, L+1, D]
            item_seq_w_mean = torch.cat([torch.ones_like(item_seq)[:, 0].view(-1, 1), item_seq], dim=1) # [B, L+1]
        else:
            x = self.sess_dropout(x)
            item_seq_w_mean = item_seq.clone()

        output, importance_alpha = self.net(item_seq_w_mean, x) # [B, L+1, D], [B, L+1, 1]

        if self.use_entmax:
            output = output * importance_alpha.to(torch.float)
            output_mask = (importance_alpha == 0)
            item_seq_w_mean[output_mask[:,:,0]] = 0
        else:
            pass

        output = F.normalize(output, dim=2) # [B, L+1, D]

        if self.extractor == 'last':
            output = output[:, 1, :].unsqueeze(1) # [B, 1, D]
            item_seq_w_mean = item_seq_w_mean[:, 0].unsqueeze(1) # [B, 1]
        if self.extractor == 'first':
            first_item_idx = torch.sum(item_seq_w_mean != 0, dim=1, keepdim=True) - 1 # [B, 1]
            output = torch.gather(output, 1, first_item_idx.unsqueeze(-1).repeat(1, 1, output.size(-1))).squeeze(1) # [B, D]
            output = output.unsqueeze(1) # [B, 1, D]
            item_seq_w_mean = item_seq_w_mean[:, 0].unsqueeze(1) # [B, 1]
        if self.extractor == 'mean':
            output = output[:, 0, :].unsqueeze(1) # [B, 1, D]
            item_seq_w_mean = item_seq_w_mean[:, 0].unsqueeze(1) # [B, 1]

        return item_seq_w_mean, output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq, output = self.forward(item_seq) # [B, L, D]
        pos_items = interaction[self.POS_ITEM_ID]

        all_item_emb = self.item_embedding.weight # [N, D]
        all_item_emb = self.item_dropout(all_item_emb) # [N, D]
        all_item_emb = F.normalize(all_item_emb, dim=-1) # [N, D] 

        logits_all = output @ all_item_emb.T # [B, 1+L, D] @ [D, N] = [B, L, N]

        if self.aggregator == 'max':
            logits = torch.max(logits_all, dim=1)[0] / self.temperature  # [B, N]  # logits[1] = [-1.4918,  0.1595, -2.9134,  ..., -1.2865,  1.9893,  2.8495],
        if self.aggregator == 'mean':
            logits_all[item_seq == 0] = 0
            logits = torch.sum(logits_all, dim=1) / torch.sum(item_seq != 0, dim=1, keepdim=True)  # [B, N]
            logits = logits / self.temperature
        if self.aggregator == 'max_mean':
            max_logits = torch.max(logits_all, dim=1)[0] / self.temperature  # [B, N]  # logits[1] = [-1.4918,  0.1595, -2.9134,  ..., -1.2865,  1.9893,  2.8495],
            
            logits_all[item_seq == 0] = 0
            mean_logits = torch.sum(logits_all, dim=1) / torch.sum(item_seq != 0, dim=1, keepdim=True)  # [B, N]
            mean_logits = mean_logits / self.temperature

            logits = max_logits * self.beta_logit + mean_logits * (1 - self.beta_logit)

        loss = self.loss_fct(logits, pos_items)

        return loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq, output = self.forward(item_seq)
        test_item_emb = self.item_embedding.weight
        test_item_emb = F.normalize(test_item_emb, dim=-1)

        scores_all = output @ test_item_emb.T # [B, L, D] @ [D, N] = [B, L, N]
        
        if self.aggregator == 'max':
            scores = torch.max(scores_all, dim=1)[0]  # [B, N]
        if self.aggregator == 'mean':
            scores_all[item_seq == 0] = 0
            scores = torch.sum(scores_all, dim=1) / torch.sum(item_seq != 0, dim=1, keepdim=True)  # [B, N]
        if self.aggregator == 'max_mean':
            max_scores = torch.max(scores_all, dim=1)[0] 

            scores_all[item_seq == 0] = 0
            mean_scores = torch.sum(scores_all, dim=1) / torch.sum(item_seq != 0, dim=1, keepdim=True)  # [B, N]

            scores = max_scores * self.beta_logit + mean_scores * (1 - self.beta_logit)

        return scores

class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        config,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            config, n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    
class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, config, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

        self.no_ffn = config['no_ffn'] if 'no_ffn' in config else True

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        if self.no_ffn:
            return attention_output
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output