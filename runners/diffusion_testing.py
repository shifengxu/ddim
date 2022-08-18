from runners.diffusion import Diffusion


class DiffusionTesting(Diffusion):
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)

    def test(self):
        pass

# class
