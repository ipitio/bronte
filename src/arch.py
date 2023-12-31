import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn
from .trainer.base import Model


class FFN(Model):
    def __init__(self):
        super(FFN, self).__init__()
        self.options["fc_layers"] = [64] * 3
        self.options["fc_dropout"] = [0.5] * 3
        self.options["num_out"] = [1] * 3

    def init_layers(self):
        input_size = len(self.X.columns)

        self.bn1 = nn.BatchNorm1d(input_size).to(self.options["device"])

        self.lin_layers = nn.ModuleList(
            [
                nn.Linear(
                    input_size if i == 0 else self.options["fc_layers"][i - 1],
                    size,
                ).to(self.options["device"])
                for i, size in enumerate(self.options["fc_layers"])
            ]
        )

        self.bn_layers = nn.ModuleList(
            [
                nn.BatchNorm1d(size).to(self.options["device"])
                for size in self.options["fc_layers"]
            ]
        )

        self.drops = nn.ModuleList(
            [
                nn.Dropout(drop).to(self.options["device"])
                for drop in self.options["fc_dropout"]
            ]
        )

        if len(self.drops) < len(self.lin_layers):
            for _ in range(len(self.lin_layers) - len(self.drops)):
                self.drops.append(nn.Dropout(0).to(self.options["device"]))

        self.out = [
            nn.Linear(
                self.options["fc_layers"][-1],
                self.options["num_out"][i],
            ).to(self.options["device"])
            for i, _ in enumerate(self.target_var)
        ]

    def forward(self, x):
        def hidden(x):
            for lin, drop, bn in zip(self.lin_layers, self.drops, self.bn_layers):
                x = F.mish(lin(x))
                if x.size(0) > 1:  # Check if batch size is greater than 1
                    x = bn(x)
                x = drop(x)
            return x

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        try:
            x = self.bn1(x)
        except:
            pass
        x = (
            checkpoint.checkpoint(hidden, x)
            if self.options["use_checkpoint"]
            else hidden(x)
        )

        m = nn.Softmax(dim=1)
        if len(self.out) > 1:
            output = []
            for i, out in enumerate(self.out):
                if self.options["num_out"][i] > 1:
                    output.append(m(out(x)))
                else:
                    output.append(out(x))
            return output  # torch.stack(output).to(self.options["device"])
        output = self.out[0](x)
        if self.options["num_out"][0] > 1:
            return m(output)
        else:
            return output


class Attention(nn.Module):
    def __init__(self, hidden_dim, method="dot", device="cpu"):
        super(Attention, self).__init__()

        self.method = method
        if method not in ["dot", "general", "concat"]:
            raise ValueError(method, "is not an appropriate attention mechanism.")

        self.hidden_dim = hidden_dim
        self.device = device
        self.hidden_transform_cache = {}

        match (method):
            case "general":
                self.attn = nn.Linear(self.hidden_dim, hidden_dim).to(device)
            case "concat":
                self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim).to(device)
                self.v = nn.Parameter(torch.FloatTensor(hidden_dim)).to(device)

    def forward(self, hidden, encoder_outputs):
        encoder_output_dim = encoder_outputs.size(2)

        # Check if we need to apply a linear transformation
        if self.hidden_dim != encoder_output_dim:
            # Check if an appropriately sized hidden_transform layer exists in the cache
            if encoder_output_dim not in self.hidden_transform_cache:
                # If not, create and cache it
                self.hidden_transform_cache[encoder_output_dim] = nn.Linear(
                    self.hidden_dim, encoder_output_dim
                ).to(self.device)
            # Use the cached layer
            hidden_transform = self.hidden_transform_cache[encoder_output_dim]
            hidden = hidden_transform(hidden)

        # Ensure hidden has the same number of dimensions as encoder_outputs
        if len(hidden.size()) < len(encoder_outputs.size()):
            hidden = hidden.unsqueeze(1)
        if hidden.size(1) != encoder_outputs.size(1):
            hidden = hidden.expand(-1, encoder_outputs.size(1), -1)

        # print("Shape of hidden:", hidden.shape)
        # print("Shape of encoder_outputs:", encoder_outputs.shape)

        # Calculate the attention weights (energies) based on the given method
        match (self.method):
            case "general":
                energy = self.attn(encoder_outputs)
                attn_energies = torch.sum(hidden * energy, dim=2)
            case "concat":
                energy = self.attn(
                    torch.cat((hidden.expand_as(encoder_outputs), encoder_outputs), 2)
                ).tanh()
                attn_energies = torch.sum(self.v * energy, dim=2)
            case _:
                attn_energies = torch.sum(hidden * encoder_outputs, dim=2)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class RNN(Model):
    def __init__(self):
        super(RNN, self).__init__()
        self.options["rnn_type"] = "GRU"
        self.options["rnn_layers"] = 1
        self.options["rnn_dropout"] = 0.5
        self.options["rnn_seq_len"] = 10
        self.options["rnn_bidi"] = False
        self.options["fc_layers"] = [64] * 3
        self.options["fc_dropout"] = [0.5] * 3
        self.options["attention"] = "dot"
        self.options["num_out"] = [1] * 3

    def init_layers(self):
        input_size = len(self.X.columns)
        rnn_type = self.options["rnn_type"]

        match (rnn_type):
            case "LSTM":
                self.rnn = nn.LSTM(
                    input_size=input_size,
                    hidden_size=input_size,
                    num_layers=self.options["rnn_layers"],
                    dropout=self.options["rnn_dropout"],
                    batch_first=True,
                    bidirectional=self.options["rnn_bidi"],
                ).to(self.options["device"])
            case "GRU":
                self.rnn = nn.GRU(
                    input_size=input_size,
                    hidden_size=input_size,
                    num_layers=self.options["rnn_layers"],
                    dropout=self.options["rnn_dropout"],
                    batch_first=True,
                    bidirectional=self.options["rnn_bidi"],
                ).to(self.options["device"])
            case _:
                raise ValueError(
                    f"Invalid RNN type: {rnn_type}. Choose either 'LSTM' or 'GRU'."
                )

        if self.options["attention"] is not None:
            self.attention = Attention(
                input_size, self.options["attention"], self.options["device"]
            ).to(self.options["device"])

        self.lin_layers = nn.ModuleList(
            [
                nn.Linear(
                    input_size * (3 if self.options["rnn_bidi"] else 2)
                    if i == 0
                    else self.options["fc_layers"][i - 1],
                    size,
                ).to(self.options["device"])
                for i, size in enumerate(self.options["fc_layers"])
            ]
        )

        self.drops = nn.ModuleList(
            [
                nn.Dropout(drop).to(self.options["device"])
                for drop in self.options["fc_dropout"]
            ]
        )

        if len(self.drops) < len(self.lin_layers):
            for _ in range(len(self.lin_layers) - len(self.drops)):
                self.drops.append(nn.Dropout(0).to(self.options["device"]))

        self.out = [
            nn.Linear(
                self.options["fc_layers"][-1],
                self.options["num_out"][i],
            ).to(self.options["device"])
            for i, _ in enumerate(self.target_var)
        ]  # Final linear layer

    def forward(self, x):
        x = x.float()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        batch_size = x.size(0)
        if self.hidden is None or not self.options["timeseries"]:
            self.hidden = self.init_hidden(batch_size)
        else:
            # Adjust the size of the hidden state to match the batch size of the input
            if self.options["rnn_type"] == "LSTM":
                h, c = self.hidden
                if h.size(1) != batch_size:
                    # Slice or pad the hidden state
                    h = self.mod_hidden(h, batch_size)
                    c = self.mod_hidden(c, batch_size)
                h, c = h.contiguous(), c.contiguous()
                self.hidden = (h, c)
            else:  # For GRU or other RNNs
                if self.hidden.size(1) != batch_size:
                    self.hidden = self.mod_hidden(self.hidden, batch_size)
                self.hidden = self.hidden.contiguous()

        # RNN Forward Pass
        if self.options["rnn_type"] == "LSTM":
            if self.options["use_checkpoint"]:
                out, (hn, cn) = (
                    checkpoint.checkpoint(self.rnn, x, self.hidden)
                    if self.hidden is not None
                    else checkpoint.checkpoint(self.rnn, x)
                )
            else:
                # Use saved states if available
                out, (hn, cn) = (
                    self.rnn(x, self.hidden) if self.hidden is not None else self.rnn(x)
                )
            self.hidden = (hn.detach(), cn.detach())
        else:  # GRU
            if self.options["use_checkpoint"]:
                out, hn = (
                    checkpoint.checkpoint(self.rnn, x, self.hidden)
                    if self.hidden is not None
                    else checkpoint.checkpoint(self.rnn, x)
                )
            else:
                out, hn = (
                    self.rnn(x, self.hidden) if self.hidden is not None else self.rnn(x)
                )
            self.hidden = hn.detach()

        # If using bidirectional RNNs, handle the forward and backward outputs
        if self.options["rnn_bidi"]:
            forward_hn = out[:, -1, : out.size(2) // 2]
            backward_hn = out[:, 0, out.size(2) // 2 :]
            rnn_out = torch.cat([forward_hn, backward_hn], dim=1)
            query = self.hidden[-2]  # For forward hidden state in a bidirectional RNN
        else:
            rnn_out = out[:, -1, :]
            query = self.hidden[-1]

        # Fully connected layers
        def fc(x):
            for lin, drop in zip(self.lin_layers, self.drops):
                x = F.mish(lin(x))
                x = drop(x)
            return x

        # attention weights and context vectors
        if self.options["attention"] is not None:

            def attn(x, query, rnn_out):
                attn_weights = self.attention(query, x)
                context = attn_weights.transpose(0, 2).bmm(x)
                context = context.squeeze(
                    1
                )  # Ensure context has shape [batch_size, hidden_dim]

                # Concatenate context vector with the last hidden state from RNN
                return torch.cat((rnn_out, context), 1)

            x = (
                checkpoint.checkpoint(attn, x, query, rnn_out)
                if self.options["use_checkpoint"]
                else attn(x, query, rnn_out)
            )

        x = checkpoint.checkpoint(fc, x) if self.options["use_checkpoint"] else fc(x)

        m = nn.Softmax(dim=1)
        if len(self.out) > 1:
            output = []
            for i, out in enumerate(self.out):
                if self.options["num_out"][i] > 1:
                    output.append(m(out(x)))
                else:
                    output.append(out(x))
            return output  # torch.stack(output).to(self.options["device"])
        output = self.out[0](x)
        if self.options["num_out"][0] > 1:
            return m(output)
        else:
            return output

    def init_hidden(self, batch_size):
        if self.options["rnn_type"] == "LSTM":
            h0 = (
                torch.zeros(
                    self.options["rnn_layers"] * (2 if self.options["rnn_bidi"] else 1),
                    batch_size,
                    self.rnn.hidden_size,
                ).to(self.options["device"])
            ).float()
            c0 = (
                torch.zeros(
                    self.options["rnn_layers"] * (2 if self.options["rnn_bidi"] else 1),
                    batch_size,
                    self.rnn.hidden_size,
                ).to(self.options["device"])
            ).float()
            return (h0, c0)
        else:  # GRU
            h0 = (
                torch.zeros(
                    self.options["rnn_layers"] * (2 if self.options["rnn_bidi"] else 1),
                    batch_size,
                    self.rnn.hidden_size,
                ).to(self.options["device"])
            ).float()
            return h0

    def mod_hidden(self, hidden, batch_size):
        if hidden.size(1) < batch_size:
            # Pad the hidden state with zeros
            padding = torch.zeros(
                hidden.size(0), batch_size - hidden.size(1), hidden.size(2)
            ).to(self.options["device"])
            return torch.cat([hidden, padding], dim=1).to(self.options["device"])
        elif hidden.size(1) > batch_size:
            # Slice the hidden state
            return hidden[:, :batch_size, :]
        else:
            return hidden
