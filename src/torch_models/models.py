
import torch
import torch.nn as nn

class LD_SIMPLE(nn.Module):
    def __init__(self,
                 h1_dim: int,
                 num_labels: int,
                 ):
        super(LD_SIMPLE, self).__init__()
        self.h1 = nn.Linear(
            in_features=num_labels,
            out_features=h1_dim
        )
        self.h_out = nn.Linear(
            in_features=h1_dim,
            out_features=num_labels
        )
        self.relu = nn.ReLU(inplace=True)

    def reset_parameters(self):
        self.h1.reset_parameters()
        self.h_out.reset_parameters()


    def forward(self, single_ld):
        """
        ld_input.shape num_matrices X dim
        """
        x = self.h1(single_ld)
        x = self.relu(x)
        x = self.h_out(x)
        return x


class LD_AVG(LD_SIMPLE):
    def __init__(self,
                 h1_dim: int,
                 num_labels: int,
                 num_neighbs: int):
        super(LD_AVG, self).__init__(
            h1_dim= h1_dim,
            num_labels = num_labels
        )
        self.num_neighbs = num_neighbs
        self.set_mparam()

    def set_mparam(self):
        #hack, change device
        self.m_weights = nn.Parameter(
            torch.as_tensor([[1. / float(self.num_neighbs)]] * self.num_neighbs, device=self.h_out.weight.device).unsqueeze(2),
            requires_grad=True
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.set_mparam()


    def forward(self, ld_input):
        """
        ld_input.shape num_matrices X dim
        """
        x = torch.sum(ld_input * self.m_weights, dim=0)
        super().forward(single_ld=x)
        return x

class LD_CONCAT(nn.Module):
    def __init__(self,
                 h1_dim: int,
                 num_labels: int,
                 num_neighbs: int):
        super().__init__()
        self.h1 = nn.Linear(
            in_features=num_labels*num_neighbs,
            out_features=h1_dim
        )
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(
            in_features=h1_dim,
            out_features=num_labels
        )

    def reset_parameters(self):
        self.h1.reset_parameters()
        self.out.reset_parameters()

    def forward(self, ld_input:torch.FloatTensor):
        inp = [t.squeeze(0) for t in ld_input.split(split_size=1, dim=0)]
        inp = torch.cat(inp, dim=1)
        x = self.h1(inp)
        x = self.relu(x)
        x = self.out(x)
        return x


class LD_INDP(nn.Module):
    def __init__(self,
                 h1_dim: int,
                 num_labels: int,
                 num_neighbs: int):
        super().__init__()
        self.h1_modules = nn.ModuleList()
        for i in range(num_neighbs):
            self.h1_modules.append(
                nn.Linear(in_features=num_labels,
                          out_features=h1_dim)
            )
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(
            in_features=h1_dim*num_neighbs,
            out_features=num_labels
        )

    def reset_parameters(self):
        for current_m in self.h1_modules:
            current_m.reset_parameters()
        self.out.reset_parameters()

    def forward(self, ld_input:torch.FloatTensor):
        inp = [t.squeeze(0) for t in ld_input.split(split_size=1, dim=0)]
        h1_outputs = [self.h1_modules[i](inp[i]) for i in range(len(inp))]
        h1 = torch.cat(h1_outputs, dim=1)
        x = self.relu(h1)
        x = self.out(x)
        return x

class LD_SHARED(nn.Module):
    def __init__(self,
                 h1_dim: int,
                 num_labels: int,
                 num_neighbs: int):
        super().__init__()
        self.h1 = nn.Linear(in_features=num_labels,
                          out_features=h1_dim)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(
            in_features=h1_dim*num_neighbs,
            out_features=num_labels
        )

    def reset_parameters(self):
        self.h1.reset_parameters()
        self.out.reset_parameters()

    def forward(self, ld_input:torch.FloatTensor):
        inp = [t.squeeze(0) for t in ld_input.split(split_size=1, dim=0)]
        h1_outputs = [self.h1(inp[i]) for i in range(len(inp))]
        h1 = torch.cat(h1_outputs, dim=1)
        x = self.relu(h1)
        x = self.out(x)
        return x