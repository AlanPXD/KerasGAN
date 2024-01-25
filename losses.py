from tensorflow import Tensor
from tensorflow import reduce_mean

def w_dist(real_disc_output:Tensor, fake_disc_output:Tensor) -> Tensor:
    """Computes the Wasserstein distance for the GAN training.

    Args:
        real_disc_output (Tensor): Discriminator output for real data
        fake_disc_output (Tensor): Discriminator output for fake data
    Returns:
        Tensor: Wasserstein distance for a batch
    """
    real_loss = reduce_mean(real_disc_output)
    fake_loss = reduce_mean(fake_disc_output)
    return real_disc_output - fake_disc_output