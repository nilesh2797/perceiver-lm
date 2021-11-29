from typing import Optional
import torch
from torch import nn

from perceiver_io import PerceiverEncoder, PerceiverDecoder, PerceiverIO, ClassificationDecoder
from perceiver_io.adapter import ImageInputAdapter, ClassificationOutputAdapter
class PerceiverLM(nn.Module):
    """Encoder-decoder based language model."""
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int,
        num_latents: int = 256,
        latent_dim: int = 1280,
        qk_out_dim = 8*32,
        v_out_dim = None,
        num_self_attn_heads=8,
        num_cross_attn_heads=8,
        num_decoder_attn_heads=8,
        self_attn_widening_factor=1,
        cross_attn_widening_factor=1,
        num_blocks=1,
        num_self_attn_per_block=12,
        dropout: float = 0.0
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim).to('cuda')
        self.position_embedding =  nn.Embedding(max_seq_len, embedding_dim).to('cuda')
        self.query_embedding = nn.Embedding(max_seq_len, embedding_dim).to('cuda')
        self.decoder_token_bias = nn.Parameter(torch.randn(vocab_size).to('cuda'))
        if v_out_dim is None: v_out_dim = latent_dim
        #input_adapter = ImageInputAdapter(
         #   image_shape=torch.squeeze(inputs).to('cuda').shape,#image_shape assuming bs = 1 which hasto be for pretrained weights
          #  num_frequency_bands=256)
        self.linear0 = nn.Linear(1568,768).to('cuda') #for 8 batch size
        self.encoder = PerceiverEncoder(
            num_latents=num_latents,
            latent_dim=latent_dim,
            input_dim=embedding_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            num_self_attn_per_block=num_self_attn_per_block,
            num_blocks=num_blocks,
            num_self_attn_heads=num_self_attn_heads,
            num_cross_attn_heads=num_cross_attn_heads,
            cross_attn_widening_factor=cross_attn_widening_factor,
            self_attn_widening_factor=self_attn_widening_factor,
            dropout=dropout,
        )
        self.decoder = PerceiverDecoder(
            #num_classes=10,
            latent_dim=latent_dim,
            query_dim=embedding_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=embedding_dim,
            num_heads=num_decoder_attn_heads,
            widening_factor=cross_attn_widening_factor,
            projection_dim=None
        )
        #self.linear1 = nn.Linear(25152,10).to('cuda') #bs = 8
        self.perceiver = PerceiverIO(self.encoder, self.decoder)
        self.linear1 = nn.Linear(25152,10).to('cuda')
        self.m = nn.Softmax(dim=1).to('cuda')        

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            inputs: Tensor of token ids.
            mask: Token mask. Mask values selected in [0, 1]. Defaults to None.
        Returns:
            Tensor of shape (batch_size, seq_len, vocab_size).
        """
        #print("after transform size is ", inputs.size())
        #print("shape input to adapter is ", torch.squeeze(inputs).to('cuda').shape)
        input_adapter = ImageInputAdapter(
            image_shape=torch.squeeze(inputs).to('cuda').shape,#image_shape assuming bs = 1 which hasto be for pretrained weights
            num_frequency_bands=256)#args.num_frequency_bands)
        #image_adapter = torch.rand(8, 96, 1568).to('cuda')
        image_adapter = input_adapter.forward(inputs)
        #print("image adapter shape is ",image_adapter.shape)
        seq_len = image_adapter.shape[1]
        fst = torch.squeeze(image_adapter).to('cuda')
        #linear0 = nn.Linear(image_adapter.shape[2],768).to('cuda')
        #print("linear0 shape", linear0)
        token_embeddings = self.linear0(fst)
        #print("linear0 ", self.linear0.weight.grad)
        if(self.linear0.weight.grad is not None):
            print("linear0 ", self.linear0.weight.grad.data.sum())  
        #print("token embeddings shape is", token_embeddings.shape)
        positions_ids = torch.arange(seq_len, device='cuda').view(1, -1).to('cuda')
        position_embeddings = self.position_embedding(positions_ids).to('cuda')
        embeddings = token_embeddings + position_embeddings
        query_embeddings = self.query_embedding(positions_ids)
        outputs = self.perceiver(
            inputs=embeddings,
            query=query_embeddings,
            input_mask=mask,
            query_mask=mask
        )
        if(self.position_embedding.weight.grad is not None):
            print("position_embedding ", self.position_embedding.weight.grad.data.sum())
        #print("output shape is", outputs.shape)
        logits = torch.matmul(outputs, self.token_embedding.weight.T).to('cuda') + self.decoder_token_bias
        #print("logits shape is", logits.shape)
        #last = logits.reshape(1,image_adapter.shape[1]*logits.shape[2]).to('cuda')
        last = logits.reshape(logits.shape[0],logits.shape[1]*logits.shape[2]).to('cuda')
        #print("last shape is", last.shape)
        #linear1 = nn.Linear(image_adapter.shape[1]*logits.shape[2],10).to('cuda')
        #linear1 = nn.Linear(logits.shape[1]*logits.shape[2],10).to('cuda')
        #print("linear1 shape ", linear1)
        output = self.linear1(last)
        if(self.linear1.weight.grad is not None):
            print("linear1 ", self.linear1.weight.grad.data.sum())
        #print(output.shape)
        #m = nn.Softmax(dim=1).to('cuda')
        #print("softmax is ",self.m)
        out = self.m(output)
        return out
