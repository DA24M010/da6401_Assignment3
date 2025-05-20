import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden)))  # (batch_size, src_len, 1)
        attention_weights = torch.softmax(score, dim=1)  # (batch_size, src_len, 1)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)  # (batch_size, hidden_size)
        return context_vector, attention_weights.squeeze(-1)

class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
        energy = torch.bmm(self.attn(encoder_outputs), hidden.transpose(1, 2)).squeeze(2)
        attn_weights = torch.softmax(energy, dim=1)  # (batch_size, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_size)
        return context, attn_weights

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_size, rnn_type="lstm", dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=0)
        self.rnn_type = rnn_type.lower()
        self.attention = LuongAttention(hidden_size)
        self.rnn = self._build_rnn(embedding_dim + hidden_size, hidden_size, dropout)
        self.fc_out = nn.Linear(hidden_size, output_dim)

    def _build_rnn(self, input_size, hidden_size, dropout):
        if self.rnn_type == 'gru':
            return nn.GRU(input_size, hidden_size, batch_first=True)
        elif self.rnn_type == 'lstm':
            return nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type")

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)  # (batch_size, 1, emb_dim)
        hidden_state = hidden[0][-1] if self.rnn_type == 'lstm' else hidden[-1]
        context_vector, attn_weights = self.attention(hidden_state, encoder_outputs)
        rnn_input = torch.cat([embedded.squeeze(1), context_vector], dim=1).unsqueeze(1)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, attn_weights

class AttentionDecoderBahdanau(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_size, rnn_type="lstm", dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=0)
        self.rnn_type = rnn_type.lower()
        self.attention = BahdanauAttention(hidden_size)
        self.rnn = self._build_rnn(embedding_dim + hidden_size, hidden_size, dropout)
        self.fc_out = nn.Linear(hidden_size, output_dim)

    def _build_rnn(self, input_size, hidden_size, dropout):
        if self.rnn_type == 'gru':
            return nn.GRU(input_size, hidden_size, batch_first=True)
        elif self.rnn_type == 'lstm':
            return nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type")

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        hidden_state = hidden[0][-1] if self.rnn_type == 'lstm' else hidden[-1]
        context_vector, attn_weights = self.attention(hidden_state, encoder_outputs)
        rnn_input = torch.cat([embedded.squeeze(1), context_vector], dim=1).unsqueeze(1)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 embedding_dim=64,
                 encoder_hidden_size=128,
                 decoder_hidden_size=128,
                 dropout=0.3,
                 rnn_type="lstm",
                 beam_size=1,
                 teacher_forcing_ratio=0.5,
                 use_attention=False,
                 attention_type="luong"):
        super().__init__()

        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=0)
        self.encoder_rnn = self._build_rnn(embedding_dim, encoder_hidden_size, dropout, rnn_type)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.rnn_type = rnn_type.lower()

        if use_attention:
            if attention_type == "luong":
                self.decoder = AttentionDecoder(target_vocab_size, embedding_dim, decoder_hidden_size, rnn_type, dropout)
            elif attention_type == "bahdanau":
                self.decoder = AttentionDecoderBahdanau(target_vocab_size, embedding_dim, decoder_hidden_size, rnn_type, dropout)
            else:
                raise ValueError("Unsupported attention type")
        else:
            self.decoder = nn.Sequential(
                nn.Embedding(target_vocab_size, embedding_dim, padding_idx=0),
                self._build_rnn(embedding_dim, decoder_hidden_size, dropout, rnn_type),
                nn.Linear(decoder_hidden_size, target_vocab_size)
            )

    def _build_rnn(self, input_size, hidden_size, dropout, rnn_type):
        if rnn_type == 'gru':
            return nn.GRU(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'lstm':
            return nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type")

    def forward(self, src, trg):
        batch_size, trg_len = trg.size()
        outputs = []
        encoder_embedded = self.encoder_embedding(src)
        encoder_outputs, hidden = self.encoder_rnn(encoder_embedded)

        input_token = trg[:, 0].unsqueeze(1)  # start token

        for t in range(1, trg_len):
            if isinstance(self.decoder, nn.Sequential):
                embedded = self.decoder[0](input_token)
                output, hidden = self.decoder[1](embedded, hidden)
                prediction = self.decoder[2](output.squeeze(1))
                attn_weights = None
            else:
                prediction, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs)
            outputs.append(prediction.unsqueeze(1))
            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            top1 = prediction.argmax(1).unsqueeze(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1

        return torch.cat(outputs, dim=1)

def build_model(input_dim, output_dim, embedding_dim=64, hidden_size=128,
                cell_type="lstm", dropout=0.3, beam_size=1,
                teacher_forcing_ratio=0.5, use_attention=False,
                attention_type="luong", device=torch.device("cpu")):
    model = Seq2Seq(
        input_vocab_size=input_dim,
        target_vocab_size=output_dim,
        embedding_dim=embedding_dim,
        encoder_hidden_size=hidden_size,
        decoder_hidden_size=hidden_size,
        dropout=dropout,
        rnn_type=cell_type,
        beam_size=beam_size,
        teacher_forcing_ratio=teacher_forcing_ratio,
        use_attention=use_attention,
        attention_type=attention_type
    ).to(device)
    return model
