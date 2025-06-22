import torch

m1 = torch.Tensor([
    [
        [1, 2],
        [3, 4],
    ],
    [
        [4, 2],
        [3, 4],
    ]
])


m2 = torch.Tensor([
    [
        [7, 2],
        [3, 4],
    ],
    [
        [2, 2],
        [3, 4],
    ]
])

print(m1 @ m2)
print(m2.exp())