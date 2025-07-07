import torch
import torch.nn as nn
import torch.optim as optim


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        features = torch.nn.functional.normalize(features, dim=1)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()
        return loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=100, feat_dim=768, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
 
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
 
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
 
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
 
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
 
        return loss


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def prompt_con_loss(prompts, labels, similarity_metric='cosine'):

    batch_size, num_prompts, prompt_dim = prompts.shape

    # [batch_size, prompt_dim_total]
    prompts_flat = prompts.view(batch_size, -1)  # 变为 [batch_size, num_prompts * prompt_dim]

    if similarity_metric == 'cosine':

        normed_prompts = F.normalize(prompts_flat, p=2, dim=-1)
        similarity_matrix = torch.matmul(normed_prompts, normed_prompts.T)  # [batch_size, batch_size]
    elif similarity_metric == 'euclidean':

        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x · y
        squared_norm = torch.sum(prompts_flat ** 2, dim=-1, keepdim=True)  # [batch_size, 1]
        similarity_matrix = squared_norm + squared_norm.T - 2 * torch.matmul(prompts_flat, prompts_flat.T)
        similarity_matrix = torch.sqrt(F.relu(similarity_matrix))  # 避免负值


    labels_expand = labels.unsqueeze(1)  # [batch_size, 1]
    same_class_mask = (labels_expand == labels_expand.T).float()  # 同类掩码 [batch_size, batch_size]
    diff_class_mask = 1 - same_class_mask  # 不同类掩码 [batch_size, batch_size]


    if similarity_metric == 'cosine':

        similarity_loss = (1 - similarity_matrix) * same_class_mask
    elif similarity_metric == 'euclidean':

        similarity_loss = similarity_matrix * same_class_mask

    similarity_loss = similarity_loss.sum() / same_class_mask.sum()  # 归一化损失


    if similarity_metric == 'cosine':

        dissimilarity_loss = F.relu(similarity_matrix) * diff_class_mask
    elif similarity_metric == 'euclidean':

        dissimilarity_loss = (-similarity_matrix) * diff_class_mask

    dissimilarity_loss = dissimilarity_loss.sum() / diff_class_mask.sum()  # 归一化损失


    total_loss = similarity_loss + dissimilarity_loss

    return total_loss


import torch
import torch.nn.functional as F


def prompt_centloss(prompts, labels, similarity_metric='cosine'):

    batch_size, num_prompts, prompt_dim = prompts.shape

    # [batch_size, prompt_dim_total]
    prompts_flat = prompts.view(batch_size, -1)  # [batch_size, num_prompts * prompt_dim]

    if similarity_metric == 'cosine':

        normed_prompts = F.normalize(prompts_flat, p=2, dim=-1)
        similarity_matrix = torch.matmul(normed_prompts, normed_prompts.T)  # [batch_size, batch_size]
    elif similarity_metric == 'euclidean':

        squared_norm = torch.sum(prompts_flat ** 2, dim=-1, keepdim=True)  # [batch_size, 1]
        similarity_matrix = squared_norm + squared_norm.T - 2 * torch.matmul(prompts_flat, prompts_flat.T)
        similarity_matrix = torch.sqrt(F.relu(similarity_matrix))  # 避免负值


    labels_expand = labels.unsqueeze(1)  # [batch_size, 1]
    same_class_mask = (labels_expand == labels_expand.T).float()  # 同类掩码 [batch_size, batch_size]


    if similarity_metric == 'cosine':

        similarity_loss = (1 - similarity_matrix) * same_class_mask
    elif similarity_metric == 'euclidean':

        similarity_loss = similarity_matrix * same_class_mask


    similarity_loss = similarity_loss.sum() / batch_size

    return similarity_loss


