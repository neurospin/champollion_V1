#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics.pairwise import rbf_kernel
from contrastive.utils import logs
import os
import matplotlib.pyplot as plt
import seaborn as sns

log = logs.set_file_logger(__file__)


def mean_off_diagonal(a):
    """Computes the mean of off-diagonal elements"""
    n = a.shape[0]
    return ((a.sum() - a.trace()) / (n * n - n))


def quantile_off_diagonal(a):
    """Computes the quantile of off-diagonal elements
    TODO: it is here the quantile of the whole a"""
    return a.quantile(0.75)


def print_info(z_i, z_j, sim_zij, sim_zii, sim_zjj, temperature):
    """prints useful info over correlations"""

    log.info("histogram of z_i after normalization:")
    log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

    log.info("histogram of z_j after normalization:")
    log.info(np.histogram(z_j.detach().cpu().numpy() * 100, bins='auto'))

    # Gives histogram of sim vectors
    log.info("histogram of sim_zij:")
    log.info(
        np.histogram(
            sim_zij.detach().cpu().numpy() *
            temperature *
            100,
            bins='auto'))

    # Diagonals as 1D tensor
    diag_ij = sim_zij.diagonal()

    # Prints quantiles of positive pairs (views from the same image)
    quantile_positive_pairs = diag_ij.quantile(0.75)
    log.info(
        f"quantile of positives ij = "
        f"{quantile_positive_pairs.cpu()*temperature*100}")

    # Computes quantiles of negative pairs
    quantile_negative_ii = quantile_off_diagonal(sim_zii)
    quantile_negative_jj = quantile_off_diagonal(sim_zjj)
    quantile_negative_ij = quantile_off_diagonal(sim_zij)

    # Prints quantiles of negative pairs
    log.info(
        f"quantile of negatives ii = "
        f"{quantile_negative_ii.cpu()*temperature*100}")
    log.info(
        f"quantile of negatives jj = "
        f"{quantile_negative_jj.cpu()*temperature*100}")
    log.info(
        f"quantile of negatives ij = "
        f"{quantile_negative_ij.cpu()*temperature*100}")
    

class BarlowTwinsLoss(nn.Module):

    def __init__(self, device, correlation='cross', lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device
        self.correlation = correlation

    def forward(self, z_a, z_b):
        # normalize repr. along the batch dimension
        # beware: normalization is not robust to batch of size 1
        # if it happens, it will return a nan loss
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)
        lbd = self.lambda_param / D

        if self.correlation=='cross':
            # cross-correlation matrix
            c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
            # loss
            c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
            # multiply off-diagonal elems of c_diff by lambda
            c_diff[~torch.eye(D, dtype=bool)] *= lbd
            loss_invariance = c_diff[torch.eye(D, dtype=bool)].sum()
            loss_redundancy = c_diff[~torch.eye(D, dtype=bool)].sum()
            loss = loss_invariance + loss_redundancy
        elif self.correlation=='auto':
            # auto-correlation matrix
            c1 = torch.mm(z_a_norm.T, z_a_norm) / N # DxD
            c2 = torch.mm(z_b_norm.T, z_b_norm) / N # DxD
            c = (c1.pow(2) + c2.pow(2)) / 2
            c[torch.eye(D, dtype=bool)]=0
            redundancy_loss = c.sum()
            # cross-correlation matrix
            c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
            # loss
            c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
            c_diff[~torch.eye(D, dtype=bool)]=0
            loss_invariance = c_diff.sum()
            loss_redundancy = self.lbd*redundancy_loss
            loss = loss_invariance + loss_redundancy
        else:
            raise ValueError("Wrong correlation specified in BarlowTwins\
                             config: use cross or auto.")

        return(loss, loss_invariance, loss_redundancy)


class VicRegLoss(nn.Module):

    def __init__(self, device=None, lmbd=5e-3, u=1, v=1, epsilon=1e-3):
        super(VicRegLoss, self).__init__()
        self.lmbd = lmbd
        self.device = device
        self.u = u
        self.v = v
        self.epsilon = epsilon

    def forward(self, x, y):
        
        bs = x.size(0)
        emb = x.size(1)

        std_x = torch.sqrt(x.var(dim=0) + self.epsilon)
        std_y = torch.sqrt(y.var(dim=0) + self.epsilon)
        var_loss = torch.mean(func.relu(1 - std_x)) + torch.mean(func.relu(1 - std_y))

        invar_loss = func.mse_loss(x, y)

        xNorm = (x - x.mean(0)) / x.std(0)
        yNorm = (y - y.mean(0)) / y.std(0)
        crossCorMat = (xNorm.T@yNorm) / bs
        cross_loss = (crossCorMat*self.lmbd - torch.eye(emb, device=torch.device('cuda'))*self.lmbd).pow(2).sum()
        
        loss = self.u*var_loss + self.v*invar_loss + cross_loss

        return loss


class CrossEntropyLoss_Classification(nn.Module):
    """
    Cross entropy loss between outputs and labels
    """

    def __init__(self, device=None, class_weights=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, output_i, output_j, labels):
        output_i = output_i.float()
        output_j = output_j.float()

        loss_i = self.loss(output_i,
                           labels[:, 0])
        loss_j = self.loss(output_j,
                           labels[:, 0])

        return 100.*(loss_i + loss_j)

    def __str__(self):
        return f"{type(self).__name__}"


class MSELoss_Regression(nn.Module):
    """
    Regression loss between outputs and regressor
    """

    def __init__(self, device=None):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, output_i, output_j, labels):
        output_i = output_i.float()
        output_j = output_j.float()
        labels = labels.float()

        loss_i = self.loss(output_i,
                           labels)
        loss_j = self.loss(output_j,
                           labels)

        return 100*(loss_i + loss_j)

    def __str__(self):
        return f"{type(self).__name__}"



import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
# import matplotlib.pyplot as plt
# import seaborn as sns


class GeneralizedSupervisedNTXenLoss(nn.Module):
    def __init__(self, kernel='rbf', temperature=0.1, return_logits=False, sigma=1.0):
        super().__init__()
        self.original_kernel = kernel  # keep for __str__
        self.sigma = sigma
        if kernel == 'rbf':
            self.kernel = lambda y1, y2: rbf_kernel(y1, y2, gamma=1. / (2 * sigma ** 2))
        else:
            assert hasattr(kernel, '__call__'), 'kernel must be a callable'
            self.kernel = kernel

        self.temperature = temperature
        self.return_logits = return_logits
        self.INF = 1e8

        # self.kernel_save_dir = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/contrastive/kernel_heatmap"
        # os.makedirs(self.kernel_save_dir, exist_ok=True)

    def forward(self, z_i, z_j, labels, return_kernel=False, step=None, writer=None):
        N = len(z_i)
        assert N == len(labels), "Unexpected labels length: %i" % len(labels)
        z_i = func.normalize(z_i, p=2, dim=-1)
        z_j = func.normalize(z_j, p=2, dim=-1)

        sim_zii = (z_i @ z_i.T) / self.temperature
        sim_zjj = (z_j @ z_j.T) / self.temperature
        sim_zij = (z_i @ z_j.T) / self.temperature

        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        all_labels = labels.view(N, -1).repeat(2, 1).detach().cpu().numpy()
        weights = self.kernel(all_labels, all_labels)
        weights = weights * (1 - np.eye(2 * N))
        weights /= weights.sum(axis=1, keepdims=True)

        # === Optional: save kernel heatmap every N steps
        # if step is not None and step % 500 == 0:
        #     plt.figure(figsize=(6, 5))
        #     sns.heatmap(weights, cmap='viridis', cbar=True)
        #     plt.title(f'RBF Kernel Similarity (σ={self.sigma})')
        #     plt.tight_layout()
        #     filename = os.path.join(self.kernel_save_dir, f"kernel_sigma{self.sigma}_step{step}.png")
        #     plt.savefig(filename)
        #     plt.close()

        sim_Z = torch.cat([
            torch.cat([sim_zii, sim_zij], dim=1),
            torch.cat([sim_zij.T, sim_zjj], dim=1)
        ], dim=0)

        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = -1. / N * (torch.from_numpy(weights).to(z_i.device) * log_sim_Z).sum()

        if self.return_logits:
            return loss, sim_zij, sim_zii, sim_zjj

        return loss

    def __str__(self):
        return f"{type(self).__name__}(temp={self.temperature}, kernel={self.original_kernel}, sigma={self.sigma})"


class CrossEntropyLoss(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations,
    arXiv 2020
    """

    def __init__(self, weights=[1, 2], reduction='sum', device=None):
        super().__init__()
        self.class_weights = torch.FloatTensor(weights).to(device)
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights,
                                        reduction=self.reduction)

    def forward(self, sample, output_i, output_j):
        sample = (sample >= 1).long()
        output_i = output_i.float()
        output_j = output_j.float()

        loss_i = self.loss(output_i,
                           sample[:, 0, :, :, :])
        loss_j = self.loss(output_j,
                           sample[:, 0, :, :, :])

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class NTXenLoss_NearestNeighbours(nn.Module):
    """
    Normalized Nearest Neighbour Temperature Cross-Entropy Loss
    for Constrastive Learning
    Refer for instance to:
    Dwibedi et al, 2021
    With a little help from my friends nearest-neighbours
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        diag_inf = self.INF * torch.eye(N, device=z_i.device)

        #####################################################
        # Computes the classical terms for NTXenLoss
        #####################################################

        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature
        sim_zji = sim_zij.T

        log.info("histogram of zij:")
        log.info(
            np.histogram(
                sim_zij.detach().cpu().numpy() *
                self.temperature,
                bins='auto'))

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_i
        #####################################################

        max_ii = torch.max(sim_zii - diag_inf, dim=1)
        max_ij = torch.max(sim_zij - diag_inf, dim=1)

        # Computes nearest-neighbour of z_i
        z_nn_i = torch.zeros(z_i.shape, device=z_i.device)
        for i in range(N):
            if max_ii.values[i] > max_ij.values[i]:
                z_nn_i[i] = z_i[max_ii.indices[i]]
            else:
                z_nn_i[i] = z_j[max_ij.indices[i]]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zii = (z_nn_i @ z_i.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zij = (z_nn_i @ z_j.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            if max_ii.values[i] > max_ij.values[i]:
                sim_nn_zii[i, max_ii.indices[i]] = -self.INF
            else:
                sim_nn_zij[i, max_ij.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zii = sim_nn_zii - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_nn_zij, sim_nn_zii], dim=1),
                                    correct_pairs)

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_j
        #####################################################

        max_jj = torch.max(sim_zjj - diag_inf, dim=1)
        max_ji = torch.max(sim_zji - diag_inf, dim=1)

        # Computes nearest-neighbour of z_j
        z_nn_j = torch.zeros(z_j.shape, device=z_j.device)
        for i in range(N):
            if max_jj.values[i] > max_ji.values[i]:
                z_nn_j[i] = z_j[max_jj.indices[i]]
            else:
                z_nn_j[i] = z_i[max_ji.indices[i]]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zjj = (z_nn_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zji = (z_nn_j @ z_i.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            if max_jj.values[i] > max_ji.values[i]:
                sim_nn_zjj[i, max_jj.indices[i]] = -self.INF
            else:
                sim_nn_zji[i, max_ji.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zjj = sim_nn_zjj - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        loss_j = func.cross_entropy(torch.cat([sim_nn_zji, sim_nn_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class NTXenLoss_WithoutHardNegative(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations,
    arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature

        # Diagonals as 1D tensor
        diag_ij = sim_zij.diagonal()

        # Prints quantiles of positive pairs (views from the same image)
        quantile_positive_pairs = diag_ij.quantile(0.75)
        log.info(
            f"quantile of positives ij = "
            f"{quantile_positive_pairs.cpu()*self.temperature*100}")

        # Computes quantiles of negative pairs
        quantile_negative_ii = quantile_off_diagonal(sim_zii)
        quantile_negative_jj = quantile_off_diagonal(sim_zjj)
        quantile_negative_ij = quantile_off_diagonal(sim_zij)

        # Prints quantiles of negative pairs
        log.info(
            f"quantile of negatives ii = "
            f"{quantile_negative_ii.cpu()*self.temperature*100}")
        log.info(
            f"quantile of negatives jj = "
            f"{quantile_negative_jj.cpu()*self.temperature*100}")
        log.info(
            f"quantile of negatives ij = "
            f"{quantile_negative_ij.cpu()*self.temperature*100}")

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        # 'Remove' the parts that are hard negatives to promote clustering
        sim_zii[sim_zii > quantile_negative_ii] = -self.INF
        sim_zjj[sim_zii > quantile_negative_jj] = -self.INF

        negative_ij = sim_zij - diag_ij.diag()
        negative_ij[negative_ij > quantile_negative_ij] = -self.INF
        negative_ij.fill_diagonal_(0.)
        sim_zij = negative_ij + diag_ij.diag()

        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1),
                                    correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class NTXenLoss_Mixed(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations,
    arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward_NearestNeighbours_OtherView(self, z_i, z_j):
        N = len(z_i)
        diag_inf = self.INF * torch.eye(N, device=z_i.device)

        #####################################################
        # Computes the classical terms for NTXenLoss
        #####################################################

        log.info("histogram of z_i before normalization:")
        log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        log.info("histogram of z_i after normalization:")
        log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature
        sim_zji = sim_zij.T

        log.info("histogram of sim_zij:")
        log.info(
            np.histogram(
                sim_zij.detach().cpu().numpy() *
                self.temperature *
                100,
                bins='auto'))

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_i
        #####################################################

        max_ii = torch.max(sim_zii - diag_inf, dim=1)
        max_ij = torch.max(sim_zij - diag_inf, dim=1)

        # Computes nearest-neighbour of z_i
        z_nn_i = torch.zeros(z_i.shape, device=z_i.device)
        z_nn_i = z_j[max_ij.indices]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zii = (z_nn_i @ z_i.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zij = (z_nn_i @ z_j.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            sim_nn_zij[i, max_ij.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zii = sim_nn_zii - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_nn_zij, sim_nn_zii], dim=1),
                                    correct_pairs)

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_j
        #####################################################

        max_jj = torch.max(sim_zjj - diag_inf, dim=1)
        max_ji = torch.max(sim_zji - diag_inf, dim=1)

        # Computes nearest-neighbour of z_j
        z_nn_j = torch.zeros(z_j.shape, device=z_j.device)
        z_nn_j = z_i[max_ji.indices]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zjj = (z_nn_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zji = (z_nn_j @ z_i.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            sim_nn_zji[i, max_ji.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zjj = sim_nn_zjj - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        loss_j = func.cross_entropy(torch.cat([sim_nn_zji, sim_nn_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def forward_NearestNeighbours(self, z_i, z_j):
        N = len(z_i)
        diag_inf = self.INF * torch.eye(N, device=z_i.device)

        #####################################################
        # Computes the classical terms for NTXenLoss
        #####################################################

        log.info("histogram of z_i before normalization:")
        log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        log.info("histogram of z_i after normalization:")
        log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature
        sim_zji = sim_zij.T

        log.info("histogram of sim_zij:")
        log.info(
            np.histogram(
                sim_zij.detach().cpu().numpy() *
                self.temperature *
                100,
                bins='auto'))

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_i
        #####################################################

        max_ii = torch.max(sim_zii - diag_inf, dim=1)
        max_ij = torch.max(sim_zij - diag_inf, dim=1)

        # Computes nearest-neighbour of z_i
        z_nn_i = torch.zeros(z_i.shape, device=z_i.device)
        for i in range(N):
            if max_ii.values[i] > max_ij.values[i]:
                z_nn_i[i] = z_i[max_ii.indices[i]]
            else:
                z_nn_i[i] = z_j[max_ij.indices[i]]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zii = (z_nn_i @ z_i.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zij = (z_nn_i @ z_j.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            if max_ii.values[i] > max_ij.values[i]:
                sim_nn_zii[i, max_ii.indices[i]] = -self.INF
            else:
                sim_nn_zij[i, max_ij.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zii = sim_nn_zii - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_nn_zij, sim_nn_zii], dim=1),
                                    correct_pairs)

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_j
        #####################################################

        max_jj = torch.max(sim_zjj - diag_inf, dim=1)
        max_ji = torch.max(sim_zji - diag_inf, dim=1)

        # Computes nearest-neighbour of z_j
        z_nn_j = torch.zeros(z_j.shape, device=z_j.device)
        for i in range(N):
            if max_jj.values[i] > max_ji.values[i]:
                z_nn_j[i] = z_j[max_jj.indices[i]]
            else:
                z_nn_j[i] = z_i[max_ji.indices[i]]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zjj = (z_nn_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zji = (z_nn_j @ z_i.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            if max_jj.values[i] > max_ji.values[i]:
                sim_nn_zjj[i, max_jj.indices[i]] = -self.INF
            else:
                sim_nn_zji[i, max_ji.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zjj = sim_nn_zjj - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        loss_j = func.cross_entropy(torch.cat([sim_nn_zji, sim_nn_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def forward_WithoutHardNegative(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature

        # Diagonals as 1D tensor
        diag_ij = sim_zij.diagonal()

        # Prints quantiles of positive pairs (views from the same image)
        quantile_positive_pairs = diag_ij.quantile(0.75)
        log.info(
            f"quantile of positives ij = "
            f"{quantile_positive_pairs.cpu()*self.temperature*100}")

        # Computes quantiles of negative pairs
        quantile_negative_ii = quantile_off_diagonal(sim_zii)
        quantile_negative_jj = quantile_off_diagonal(sim_zjj)
        quantile_negative_ij = quantile_off_diagonal(sim_zij)

        # Prints quantiles of negative pairs
        log.info(
            f"quantile of negatives ii = "
            f"{quantile_negative_ii.cpu()*self.temperature*100}")
        log.info(
            f"quantile of negatives jj = "
            f"{quantile_negative_jj.cpu()*self.temperature*100}")
        log.info(
            f"quantile of negatives ij = "
            f"{quantile_negative_ij.cpu()*self.temperature*100}")

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        # 'Remove' the parts that are hard negatives to promote clustering
        sim_zii[sim_zii > quantile_negative_ii] = -self.INF
        sim_zjj[sim_zjj > quantile_negative_jj] = -self.INF

        # 'Remove' the parts that are hard negatives to promote clustering
        # We keep the positive element j (second view)
        negative_ij = sim_zij - diag_ij.diag()
        negative_ij[negative_ij > quantile_negative_ij] = -self.INF
        negative_ij.fill_diagonal_(0.)
        sim_zij = negative_ij + diag_ij.diag()

        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1),
                                    correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def forward(self, z_i, z_j):
        loss_NN, _, _ = self.forward_NearestNeighbours_OtherView(z_i, z_j)
        loss_WHN, sim_zij, correct_pairs = self.forward_WithoutHardNegative(
            z_i, z_j)

        if self.return_logits:
            return ((loss_NN + loss_WHN) / 2), sim_zij, correct_pairs

        return ((loss_NN + loss_WHN) / 2)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)
