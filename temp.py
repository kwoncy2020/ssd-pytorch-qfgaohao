
import torch

yv, xv = torch.meshgrid([torch.arange(5), torch.arange(3)])
print(yv)
print(xv)
print('yv shape:', yv.shape, 'xv shape:', xv.shape)
grid = torch.stack((xv,yv),2)
print(grid.shape)
grid2 = grid.view(1,1,5,3,2)
print(grid2.shape)
