import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Seq2Seq(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        target_vocab_size,
        embedding_dim=64,
        encoder_hidden_size=128,
        decoder_hidden_size=128,
        encoder_num_layers=2,
        decoder_num_layers=2,
        dropout=0.3,
        rnn_type="gru",
        beam_size=1,
        teacher_forcing_ratio=0.5  # New parameter
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.rnn_type = rnn_type.lower()
        self.beam_size = beam_size
        self.teacher_forcing_ratio = teacher_forcing_ratio  # Store teacher forcing ratio

        # Embedding layers
        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=0)
        self.decoder_embedding = nn.Embedding(target_vocab_size, embedding_dim, padding_idx=0)

        # RNN layers
        self.encoder_rnn = self._build_rnn(embedding_dim, encoder_hidden_size, encoder_num_layers, dropout)
        self.decoder_rnn = self._build_rnn(embedding_dim, decoder_hidden_size, decoder_num_layers, dropout)

        # Final output layer
        self.fc_out = nn.Linear(decoder_hidden_size, target_vocab_size)
        
        # Add hidden state adapter if encoder and decoder have different numbers of layers
        if encoder_num_layers != decoder_num_layers:
            if self.rnn_type == 'lstm':
                self.hidden_adapter = nn.ModuleList([
                    nn.Linear(encoder_hidden_size, decoder_hidden_size),  # For hidden state
                    nn.Linear(encoder_hidden_size, decoder_hidden_size)   # For cell state
                ])
            else:
                self.hidden_adapter = nn.Linear(encoder_hidden_size, decoder_hidden_size)

    def _build_rnn(self, input_size, hidden_size, num_layers, dropout):
        if self.rnn_type == 'rnn':
            return nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'gru':
            return nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'lstm':
            return nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unsupported RNN cell type: {self.rnn_type}")

    def _adapt_hidden_state(self, encoder_hidden):
        """Adapt encoder hidden state to decoder hidden state dimensions"""
        if self.encoder_num_layers == self.decoder_num_layers:
            return encoder_hidden
            
        if self.rnn_type == 'lstm':
            # LSTM has a tuple of (hidden_state, cell_state)
            h, c = encoder_hidden
            
            # Take the last layer's hidden state and cell state
            if self.encoder_num_layers > self.decoder_num_layers:
                # If encoder has more layers, take the last decoder_num_layers
                h = h[-self.decoder_num_layers:]
                c = c[-self.decoder_num_layers:]
            else:
                # If decoder has more layers, expand the hidden state
                batch_size = h.size(1)
                h_adapted = self.hidden_adapter[0](h[-1])
                c_adapted = self.hidden_adapter[1](c[-1])
                
                # Repeat the adapted hidden state for all decoder layers
                h = h_adapted.unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
                c = c_adapted.unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
                
            return (h, c)
        else:
            # RNN or GRU has a single hidden state tensor
            if self.encoder_num_layers > self.decoder_num_layers:
                # If encoder has more layers, take the last decoder_num_layers
                return encoder_hidden[-self.decoder_num_layers:]
            else:
                # If decoder has more layers, expand the hidden state
                batch_size = encoder_hidden.size(1)
                h_adapted = self.hidden_adapter(encoder_hidden[-1])
                
                # Repeat the adapted hidden state for all decoder layers
                return h_adapted.unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)

    def forward(self, src, tgt, teacher_forcing_ratio=None):
        """
        Forward pass with teacher forcing
        
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            teacher_forcing_ratio: Probability of using teacher forcing.
                                  If None, use self.teacher_forcing_ratio
        """
        # Use default teacher_forcing_ratio if not provided
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio
            
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.fc_out.out_features
        device = src.device
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(device)
        
        # Encoder forward pass
        embedded_src = self.encoder_embedding(src)
        encoder_outputs, encoder_hidden = self.encoder_rnn(embedded_src)
        
        # Adapt hidden state dimensions if needed
        hidden = self._adapt_hidden_state(encoder_hidden)
        
        # First input to the decoder is the <sos> token
        input = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            # Get the embedding of the current input
            embedded_input = self.decoder_embedding(input)
            
            # Forward pass through decoder
            output, hidden = self.decoder_rnn(embedded_input, hidden)
            
            # Get prediction
            prediction = self.fc_out(output.squeeze(1))
            
            # Store prediction
            outputs[:, t-1] = prediction
            
            # Decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token from our predictions
            top1 = prediction.argmax(1)
            
            # If teacher forcing, use actual next token as next input
            # If not, use predicted token
            input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        # Final step
        embedded_input = self.decoder_embedding(input)
        output, hidden = self.decoder_rnn(embedded_input, hidden)
        prediction = self.fc_out(output.squeeze(1))
        outputs[:, -1] = prediction
            
        return outputs

    def predict(self, src, sos_token, eos_token, max_len=50):
        """
        Generate predictions without teacher forcing
        """
        self.eval()
        batch_size = src.size(0)
        
        # Encoder
        embedded_src = self.encoder_embedding(src)
        encoder_outputs, encoder_hidden = self.encoder_rnn(embedded_src)
        
        # Adapt hidden state
        hidden = self._adapt_hidden_state(encoder_hidden)
        
        # Beam search
        beams = [([sos_token], 0.0, hidden)]

        for _ in range(max_len):
            new_beams = []
            for seq, score, hidden in beams:
                last_token = torch.tensor([[seq[-1]]], device=src.device)
                embedded = self.decoder_embedding(last_token)
                output, hidden = self.decoder_rnn(embedded, hidden)
                logits = self.fc_out(output.squeeze(1))
                probs = F.log_softmax(logits, dim=1)

                topk_probs, topk_ids = probs.topk(self.beam_size)
                for i in range(self.beam_size):
                    new_seq = seq + [topk_ids[0][i].item()]
                    new_score = score + topk_probs[0][i].item()
                    new_beams.append((new_seq, new_score, hidden))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.beam_size]
            if all(seq[-1] == eos_token for seq, _, _ in beams):
                break

        return beams[0][0]  # best sequence

def build_model(input_dim, output_dim, embedding_dim=64, hidden_size=128,
                encoder_layers=1, decoder_layers=1, cell_type="lstm", dropout=0.3,
                beam_size=1, teacher_forcing_ratio=0.5, device=torch.device("cpu")):
    """
    Create a new Seq2Seq model instance with the specified parameters
    """
    model = Seq2Seq(
        input_vocab_size=input_dim,
        target_vocab_size=output_dim,
        embedding_dim=embedding_dim,
        encoder_hidden_size=hidden_size,
        decoder_hidden_size=hidden_size,
        encoder_num_layers=encoder_layers,
        decoder_num_layers=decoder_layers,
        dropout=dropout,
        rnn_type=cell_type,
        beam_size=beam_size,
        teacher_forcing_ratio=teacher_forcing_ratio
    ).to(device)
    return model