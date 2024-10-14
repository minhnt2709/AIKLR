import torch
from torch import nn


class AMRInfusionMul(nn.Module):
    def __init__(
        self,
        text_model,
        amr_model,
        dropout: float,
        concat_emb_dim=int,
        emb_dim=int,
        mode=str,
        infusion_type=None,
        batch_size=int,
        device=str,
    ):
        super().__init__()
        self.text_model = text_model
        self.amr_model = amr_model
        self.mode = mode
        self.infusion_type = infusion_type
        self.batch_size = batch_size
        self.device = device

        self.linear = nn.Linear(concat_emb_dim, emb_dim)
        self.amr_linear = nn.Linear(768 * 2, 768)
        self.linear_mul = nn.Linear(768, 1)
        self.dropout_layer = nn.Dropout(dropout)
        self.final_linear = nn.Linear(emb_dim, 2)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=8, dropout=0.1
        )

    def forward(self, amr_sample, text_input):
        text_input_ids = text_input["input_ids"]
        text_attention_mask = text_input["attention_mask"]
        text_token_type_ids = text_input["token_type_ids"]

        text_input_ids = (
            text_input_ids.view(-1, text_input_ids.size(-1)).to(self.device)
            if text_input_ids is not None
            else None
        )

        text_attention_mask = (
            text_attention_mask.view(-1, text_attention_mask.size(-1)).to(self.device)
            if text_attention_mask is not None
            else None
        )
        text_token_type_ids = (
            text_token_type_ids.view(-1, text_token_type_ids.size(-1)).to(self.device)
            if text_token_type_ids is not None
            else None
        )

        # text model: cross encoder (question, article)
        text_embedding = self.text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
        )
        text_last_hidden_state = text_embedding.last_hidden_state
        text_embedding = text_embedding.pooler_output
        
        # print(text_embedding.shape)
        qamr_embedding, aamr_embedding = self.get_amr_embedding(
            amr_sample, self.amr_model, self.batch_size
        )
        # print(qamr_embedding[0])

        if self.mode == "infusion":
            amr_embedding = torch.cat((qamr_embedding, aamr_embedding), axis=1).to(self.device)
            amr_embedding = self.dropout_layer(amr_embedding)
            amr_embedding = self.amr_linear(amr_embedding)
            amr_embedding = torch.tanh(amr_embedding)
            amr_embedding = self.dropout_layer(amr_embedding)

            if self.infusion_type == "concat":
                # shallow infusion by concatenating two embeddings
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
                )  # 1x768
                amr_text_attn, w_attn = self.cross_attn(
                    query=amr_embedding,
                    key=text_embedding,
                    value=text_embedding,
                    need_weights=False,
                )  # 1x768
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

                concat_embedding = torch.cat((text_amr_attn, amr_text_attn), axis=1)
                concat_embedding = self.linear(concat_embedding)
                concat_embedding = torch.tanh(concat_embedding)
                concat_embedding = self.dropout_layer(concat_embedding)
                logits = self.final_linear(concat_embedding)
                
            elif self.infusion_type == "co_attn_res_v2":
                amr_last_hidden_state = torch.cat((text_embedding, amr_embedding), axis=0)
            
                text_amr_attn, w_attn = self.cross_attn(
                    query=text_embedding,
                    key=amr_last_hidden_state,
                    value=amr_last_hidden_state,
                    need_weights=False,
                )  
                amr_text_attn, w_attn = self.cross_attn(
                    query=amr_embedding,
                    key=text_last_hidden_state.squeeze(),
                    value=text_last_hidden_state.squeeze(),
                    need_weights=False,
                )  
                
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

                concat_embedding = torch.cat((text_amr_attn, amr_text_attn), axis=1)
                concat_embedding = self.linear(concat_embedding)
                concat_embedding = torch.tanh(concat_embedding)
                concat_embedding = self.dropout_layer(concat_embedding)
                logits = self.final_linear(concat_embedding)
                
        elif self.mode == "only_text":
            text_embedding = self.dropout_layer(text_embedding)
            text_embedding = self.final_linear(text_embedding)
            logits = text_embedding

        elif self.mode == "only_amr":
            amr_embedding = torch.cat((qamr_embedding, aamr_embedding), axis=1).to(self.device)
            amr_embedding = self.amr_linear(amr_embedding)
            amr_embedding = torch.tanh(amr_embedding)
            amr_embedding = self.dropout_layer(amr_embedding)
            amr_embedding = self.final_linear(amr_embedding)
            logits = amr_embedding

        return logits

    # input: amr sample: class with attribute guid, sentence, edge_index, edge_type, pos_ids, err_flag
    # if amr_sample.err_flag = True then amr feature is missing
    # question_id: id of the question to find missing amr sample
    # amr_model: amr model load from pretrained path
    def get_amr_embedding(self, amr_sample, amr_model, batch_size):
        err_flag = amr_sample.err_flag

        qamr_emb = torch.zeros([1, 768], dtype=torch.float32)
        aamr_emb = torch.zeros([1, 768], dtype=torch.float32)
        if err_flag == False:
            qsent = amr_sample.texts[0]
            qedge_index = amr_sample.edge_index[0]
            qedge_type = amr_sample.edge_type[0]
            qpos_ids = amr_sample.pos_ids[0]    

            asent = amr_sample.texts[1]
            aedge_index = amr_sample.edge_index[1]
            aedge_type = amr_sample.edge_type[1]
            apos_ids = amr_sample.pos_ids[1]

            qamr_emb = amr_model.encode(
                [qsent],
                graph_index=[qedge_index],
                graph_type=[qedge_type],
                pos_ids=[qpos_ids],
                batch_size=batch_size,
                convert_to_tensor=True,
            )
            aamr_emb = amr_model.encode(
                [asent],
                graph_index=[aedge_index],
                graph_type=[aedge_type],
                pos_ids=[apos_ids],
                batch_size=batch_size,
                convert_to_tensor=True,
            )

        return qamr_emb, aamr_emb