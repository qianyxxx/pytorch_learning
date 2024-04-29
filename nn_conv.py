import torch
import torch.nn.functional as F

# 5x5 input
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 3x3 kernel
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5)) # 1: batch_size 1: channel 5: height 5: width
kernel = torch.reshape(kernel, (1, 1, 3, 3)) # 1: output_channel 1: input_channel 3: kernel_height 3: kernel_width

print(input.shape)  # torch.Size([1, 1, 5, 5])
print(kernel.shape) # torch.Size([1, 1, 3, 3])

output = F.conv2d(input, kernel, stride=1, padding=0) # stride: 步长 padding: 填充
print(output)
print(output.shape) # torch.Size([1, 1, 3, 3])

output_2 = F.conv2d(input, kernel, stride=2, padding=0) # stride: 步长 padding: 填充
print(output_2)
print(output_2.shape) # torch.Size([1, 1, 2, 2])

output_3 = F.conv2d(input, kernel, stride=1, padding=1) # stride: 步长 padding: 填充
print(output_3)
print(output_3.shape) # torch.Size([1, 1, 5, 5])