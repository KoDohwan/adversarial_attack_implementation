import torch
import torch.nn as nn
from resnet import ResNet18

class FGSM():
    def __init__(self, epsilon, device):
        self.epsilon = epsilon
        self.device = device

    def __call__(self, batch, label, model, loss_function):
        batch = batch.clone().to(self.device)
        label = label.clone().to(self.device)
        batch.requires_grad = True

        output = model(batch)
        loss = loss_function(output, label)

        grad = torch.autograd.grad(loss, batch)[0]

        adv_batch = batch + self.epsilon * grad.sign()
        adv_batch = torch.clamp(adv_batch, min=0, max=255)
        return adv_batch

class IFGSM():
    def __init__(self, alpha, epsilon, n_iteration, device):
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_iteration = n_iteration
        self.device = device

    def __call__(self, batch, label, model, loss_function):
        batch = batch.clone().to(self.device)
        label = label.clone().to(self.device)

        adv_batch = batch
        for i in range(self.n_iteration):
            adv_batch.requires_grad = True

            output = model(adv_batch)
            loss = loss_function(output, label)

            grad = torch.autograd.grad(loss, adv_batch)[0]

            adv_batch.requires_grad = False
            adv_batch += self.alpha * grad.sign()

            diff = adv_batch - batch
            diff = torch.clamp(diff, min=-self.epsilon, max=self.epsilon)
            adv_batch = batch + diff

            adv_batch = torch.clamp(adv_batch, min=0, max=255)
        return adv_batch

class LLFGSM():
    def __init__(self, alpha, epsilon, n_iteration, device):
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_iteration = n_iteration
        self.device = device

    def __call__(self, batch, label, model, loss_function):
        batch = batch.clone().to(self.device)
        label = label.clone().to(self.device)

        adv_batch = batch
        for i in range(self.n_iteration):
            adv_batch.requires_grad = True

            output = model(adv_batch)
            LL_output = output.min(1)[1]

            loss = loss_function(output, LL_output)

            grad = torch.autograd.grad(loss, adv_batch)[0]

            adv_batch.requires_grad = False
            adv_batch -= self.alpha * grad.sign()

            diff = adv_batch - batch
            diff = torch.clamp(diff, min=-self.epsilon, max=self.epsilon)
            adv_batch = batch + diff

            adv_batch = torch.clamp(adv_batch, min=0, max=255)
        return adv_batch

if __name__ == '__main__':
    model = ResNet18()
    loss_function = nn.CrossEntropyLoss()
    batch = torch.randn((8, 3, 28, 28))
    label = torch.randint(low=0, high=9, size=(8,))

    # atk = FGSM(epsilon=4, device='cpu')
    # atk = IFGSM(alpha=4, epsilon=30, n_iteration=10, device='cpu')
    atk = LLFGSM(alpha=4, epsilon=30, n_iteration=10, device='cpu')

    print(atk(batch, label, model, loss_function).shape)