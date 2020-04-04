import torch.nn.functional as F
import torch.nn as nn
import torch


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        

class SiameseDeepLSTMNet(nn.Module):
    """
    Seamise LSTM network for learning acoustic word embeddings (AWEs).
    """

    def __init__(self, model_settings):
        super(SiameseDeepLSTMNet, self).__init__()
        self.__n_window_height = model_settings['input_dim']
        self.__n_classes = model_settings['label_count']

        self.__dropout = nn.Dropout(p=model_settings['dropout'])

        if 'batch_norm' in model_settings and model_settings['batch_norm'] is True:
            self.__bn1 = nn.BatchNorm1d(self.__n_window_height)

        self.__n_hidden_reccurent = model_settings['hidden_reccurent']
        self.__n_hidden_fc = model_settings['hidden_fc']
        if 'bidirectional' in model_settings:
            bidirectional = model_settings['bidirectional']
        else:
            bidirectional = False
        k = 2 if bidirectional else 1
        self.lstms = nn.ModuleList([nn.LSTM(self.__n_window_height, self.__n_hidden_reccurent[0], batch_first=True,
                                            bidirectional=bidirectional)])
        self.lstms.extend(
            [nn.LSTM(self.__n_hidden_reccurent[i - 1] * k, self.__n_hidden_reccurent[i], batch_first=True,
                     bidirectional=bidirectional)
             for i in range(1, len(self.__n_hidden_reccurent))])

        if self.__n_hidden_fc is not None:
            self.linears = nn.ModuleList(
                [nn.Linear(self.__n_hidden_reccurent[-1] * k, self.__n_hidden_fc[0])])
            self.linears.extend(
                [nn.Linear(self.__n_hidden_fc[i - 1], self.__n_hidden_fc[i]) for i in
                 range(1, len(self.__n_hidden_fc))])

            self.__output_layer = nn.Linear(self.__n_hidden_fc[-1] * 2, 1)
            self.__output_layer_cce = nn.Linear(self.__n_hidden_fc[-1], self.__n_classes)
        else:
            self.__output_layer = nn.Linear(self.__n_hidden_reccurent[-1] * 2 * k, 1)
            self.__output_layer_cce = nn.Linear(self.__n_hidden_reccurent[-1] * k, self.__n_classes)

        # self.apply(init_weights)

    def single_forward(self, input):
        """Compute embeddings of the input tensor

        Parameters
        ----------
        input: torch.Tensor
            input tensor
        Returns
        -------
        torch.Tensor:
            output embeddings computed over all timesteps of the input tensor
        torch.Tensor
            embedding of the last timestep

        """
        x = input
        if hasattr(self, '__bn1'):
            orig_shape = x.shape
            x = x.view(-1, self.__n_window_height)
            x = self.__bn1(x)
            x = x.view(orig_shape)
        x = self.__dropout(x)

        for i in range(len(self.__n_hidden_reccurent) - 1):
            x, hidden = self.lstms[i](x, None)
            x = self.__dropout(x)
        x, hidden = self.lstms[-1](x, None)

        if self.__n_hidden_fc is not None:
            for i in range(len(self.__n_hidden_fc) - 1):
                x = torch.relu(self.linears[i](x))
                x = self.__dropout(x)
            # x = torch.tanh(self.linears[-1](x))
            x = self.linears[-1](x)
        hidden = x[:, -1, :]
        return x, hidden

    def forward(self, input, hidden=None):
        """Compute embeddings of the input, where input is a pair of tensors according to Seamise approach.
        Output embeddings as well as predicts logits for class predictions.

        Parameters
        ----------
        input: list(torch.Tensor)
            Pair of input tensors
        hidden: optinal

        Returns
        -------
        list(torch.Tensor)
            Pair of output embeddings
        torch.Tensor
            Logits
        """
        lstm_out = []
        embeddings = []
        for i in range(2):
            x = input[i]
            x, hidden = self.single_forward(x)
            lstm_out.append(hidden)
            embeddings.append(x)

        # cce path
        cce_output = torch.cat(lstm_out, dim=0).squeeze()
        cce_output = self.__output_layer_cce(cce_output)
        return embeddings, cce_output