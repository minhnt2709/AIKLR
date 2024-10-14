import torch
from torch import nn


class AMRBARTInfusion(nn.Module):
    def __init__(
        self,
        text_model,
        amr_model,
        dropout: float,
        concat_emb_dim=int,
        emb_dim=int,
        mode=str,
        infusion_type=None,
        amr_eos_token_id=str,
    ):
        super().__init__()
        self.text_model = text_model
        self.amr_model = amr_model
        self.amr_eos_token_id = amr_eos_token_id
        self.mode = mode
        self.infusion_type = infusion_type

        self.linear = nn.Linear(concat_emb_dim, emb_dim)
        self.amr_linear = nn.Linear(768 * 2, 768)
        self.dropout_layer = nn.Dropout(dropout)
        self.final_linear = nn.Linear(emb_dim, 2)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=8, dropout=0.1
        )

    def forward(
        self,
        amr_input,
        text_input
    ):
        text_input_ids = text_input["input_ids"]
        text_attention_mask = text_input["attention_mask"]
        text_token_type_ids = text_input["token_type_ids"]

        text_input_ids = (
            text_input_ids.view(-1, text_input_ids.size(-1)).to("cuda")
            if text_input_ids is not None
            else None
        )
        text_attention_mask = (
            text_attention_mask.view(-1, text_attention_mask.size(-1)).to("cuda")
            if text_attention_mask is not None
            else None
        )
        text_token_type_ids = (
            text_token_type_ids.view(-1, text_token_type_ids.size(-1)).to("cuda")
            if text_token_type_ids is not None
            else None
        )

        amr_input_ids = torch.tensor(amr_input["input_ids"], dtype=torch.int)
        amr_attention_mask = torch.tensor(amr_input["attention_mask"], dtype=torch.int)

        amr_input_ids = (
            amr_input_ids.view(-1, amr_input_ids.size(-1)).to("cuda")
            if amr_input_ids is not None
            else None
        )

        amr_attention_mask = (
            amr_attention_mask.view(-1, amr_attention_mask.size(-1)).to("cuda")
            if amr_attention_mask is not None
            else None
        )

        # text model: cross encoder (question, article)
        text_embedding = self.text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
        )[1]

        amr_last_hidden_state = self.amr_model(
            input_ids=amr_input_ids, attention_mask=amr_attention_mask
        )[0]

        eos_mask = amr_input_ids.eq(self.amr_eos_token_id)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        amr_embedding = amr_last_hidden_state[eos_mask, :].view(
            amr_last_hidden_state.size(0), -1, amr_last_hidden_state.size(-1)
        )[:, -1, :]
        
        if self.mode == "only_amr":       
            amr_embedding = self.dropout_layer(amr_embedding)
            amr_embedding = self.final_linear(amr_embedding)
            logits = amr_embedding
        elif self.mode == "infusion":
            if self.infusion_type == "concat":
                concat_embedding = torch.cat((text_embedding, amr_embedding), axis=1)
                concat_embedding = self.linear(concat_embedding)
                concat_embedding = torch.tanh(concat_embedding)
                concat_embedding = self.dropout_layer(concat_embedding)
                logits = self.final_linear(concat_embedding)
            elif self.infusion_type == "co_attn_res":
                text_amr_attn, w_attn = self.cross_attn(
                    query=text_embedding,
                    key=amr_embedding,
                    value=amr_embedding,
                    need_weights=False,
                ) #1x768
                amr_text_attn, w_attn = self.cross_attn(
                    query=amr_embedding,
                    key=text_embedding,
                    value=text_embedding,
                    need_weights=False,
                ) #1x768
                text_amr_attn = self.dropout_layer(text_amr_attn)
                amr_text_attn = self.dropout_layer(amr_text_attn)
                
                text_amr_attn = torch.cat((text_amr_attn, text_embedding), axis=1)
                amr_text_attn = torch.cat((amr_text_attn, amr_embedding), axis=1)
                text_amr_attn = self.linear(text_amr_attn)
                amr_text_attn = self.linear(amr_text_attn)
                
                text_amr_attn = torch.tanh(text_amr_attn)
                text_amr_attn = self.dropout_layer(text_amr_attn)
                amr_text_attn = torch.tanh(amr_text_attn)
                amr_text_attn = self.dropout_layer(amr_text_attn)
                
                concat_embedding = torch.cat((text_amr_attn, amr_text_attn),axis=1)
                concat_embedding = self.linear(concat_embedding)
                concat_embedding = torch.tanh(concat_embedding)
                concat_embedding = self.dropout_layer(concat_embedding)
                logits = self.final_linear(concat_embedding)
                
        return logits
