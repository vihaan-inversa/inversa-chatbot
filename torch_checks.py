import torch
print(torch.__version__)
print(torch.backends.mps.is_available())  # Should return True

import torch
print(hasattr(torch, "compiler"))  # Should be True in 2.1.0+

