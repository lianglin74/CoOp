import torch


class TensorQueue(torch.nn.Module):
    def __init__(self, N, dim):
        super().__init__()

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue", torch.randn(N, dim))
        self.N = N
        self.dim = dim

        self.queue = torch.nn.functional.normalize(self.queue, dim=1)

    @torch.no_grad()
    def en_de_queue(self, data):
        if self.N == 0:
            return
        # forward means enqueue and dequeue
        data = data.detach()
        batch_size = data.shape[0]

        ptr = int(self.queue_ptr)

        assert self.N % batch_size == 0
        assert data.shape[1] == self.dim
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = data
        ptr = (ptr + batch_size) % self.N  # move pointer
        self.queue_ptr[0] = ptr

    def get_order(self, batch_size):
        ptr = int(self.queue_ptr)
        c = ptr // batch_size
        total = self.N // batch_size
        first = torch.arange(c, 0, -1)
        second = torch.arange(total, c, -1)
        order = torch.cat([first, second])
        return order
