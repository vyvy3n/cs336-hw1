
import torch 
from typing import Optional
from collections.abc import Callable, Iterable



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid Learning Rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)  
                    state['v'] = torch.zeros_like(p.data)  
                
                state['step'] += 1
                step = state['step']
                
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * (grad * grad)
                
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                adjusted_lr = lr * (bias_correction2 ** 0.5) / bias_correction1
                p.data = p.data - adjusted_lr * state['m'] / (torch.sqrt(state['v']) + eps)
                if weight_decay > 0:
                    p.data = p.data - lr * weight_decay * p.data
        
        return loss


if __name__ == "__main__":
    # Simple test
    torch.manual_seed(42)
    
    # Create a simple linear layer
    model = torch.nn.Linear(10, 1)
    target = torch.randn(5, 1)
    
    # Test our AdamW
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    
    for i in range(3):
        x = torch.randn(5, 10)
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Step {i+1}: Loss = {loss.item():.6f}")
    
    print("AdamW test completed successfully!")