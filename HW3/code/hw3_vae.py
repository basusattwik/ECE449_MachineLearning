import sys
import argparse
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import matplotlib.image
from matplotlib.pyplot import figure
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from hw3_utils import array_to_image, concat_images, batch_indices, load_mnist




# The "encoder" model q(z|x)
class Encoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Encoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units
        
        self.fc1 = nn.Linear(data_dimension, hidden_units)
        self.fc2_mu = nn.Linear(hidden_units, latent_dimension)
        self.fc2_sigma = nn.Linear(hidden_units, latent_dimension)

    def forward(self, x):
        # Input: x input image [batch_size x data_dimension]
        # Output: parameters of a diagonal gaussian 
        #   mean : [batch_size x latent_dimension]
        #   variance : [batch_size x latent_dimension]

        hidden = torch.tanh(self.fc1(x))
        mu = self.fc2_mu(hidden)
        log_sigma_square = self.fc2_sigma(hidden)
        sigma_square = torch.exp(log_sigma_square)  
        return mu, sigma_square


# "decoder" Model p(x|z)
class Decoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Decoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units

        # TODO: define the parameters of the decoder
        # fc1: a fully connected layer with 500 hidden units. 
        # fc2: a fully connected layer with 500 hidden units. 
        # nl1, nl2: nonlinearities 

        self.fc1 = nn.Linear(in_features=latent_dimension, out_features=hidden_units)
        self.fc2 = nn.Linear(in_features=hidden_units, out_features=data_dimension)

    def forward(self, z):
        # input
        #   z: latent codes sampled from the encoder [batch_size x latent_dimension]
        # output 
        #   p: a tensor of the same size as the image indicating the probability of every pixel being 1 [batch_size x data_dimension]

        # TODO: implement the decoder here. The decoder is a multi-layer perceptron with two hidden layers. 
        # The first layer is followed by a tanh non-linearity and the second layer by a sigmoid.
        
        x1 = self.fc1(z)
        x2 = torch.tanh(x1)
        x3 = self.fc2(x2)
        p  = torch.sigmoid(x3)

        return p


# VAE model
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dimension = args.latent_dimension
        self.hidden_units =  args.hidden_units
        self.data_dimension = args.data_dimension
        self.resume_training = args.resume_training
        self.batch_size = args.batch_size
        self.num_epoches = args.num_epoches
        self.e_path = args.e_path
        self.d_path = args.d_path

        # load and pre-process the data
        N_data, self.train_images, self.train_labels, test_images, test_labels = load_mnist()

        # Instantiate the encoder and decoder models 
        self.encoder = Encoder(self.latent_dimension, self.hidden_units, self.data_dimension)
        self.decoder = Decoder(self.latent_dimension, self.hidden_units, self.data_dimension)

        # Load the trained model parameters
        if self.resume_training:
            self.encoder.load_state_dict(torch.load(self.e_path))
            self.decoder.load_state_dict(torch.load(self.d_path))

    # Sample from Diagonal Gaussian z~N(μ,σ^2 I) 
    @staticmethod
    def sample_diagonal_gaussian(mu, sigma_square):
        # Inputs:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   sample: from a diagonal gaussian with mean mu and variance sigma_square [batch_size x latent_dimension]

        # TODO: Implement the reparameterization trick and return the sample z [batch_size x latent_dimension]
        epsilon = torch.randn(mu.size(dim=0), mu.size(dim=1))
        sample  = mu + torch.sqrt(sigma_square) * epsilon # reparameterized z

        return sample

    # Sampler from Bernoulli
    @staticmethod
    def sample_Bernoulli(p):
        # Input: 
        #   p: the probability of pixels labeled 1 [batch_size x data_dimension]
        # Output:
        #   x: pixels'labels [batch_size x data_dimension], type should be torch.float32

        # TODO: Implement a sampler from a Bernoulli distribution
        x = torch.bernoulli(p)

        return x


    # Compute Log-pdf of z under Diagonal Gaussian N(z|μ,σ^2 I)
    @staticmethod
    def logpdf_diagonal_gaussian(z, mu, sigma_square):
        # Input:
        #   z: sample [batch_size x latent_dimension]
        #   mu: mean of the gaussian distribution [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian distribution [batch_size x latent_dimension]
        # Output:
        #    logprob: log-probability of a diagonal gaussian [batch_size]
        
        # TODO: implement the logpdf of a gaussian with mean mu and variance sigma_square*I
        normal  = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=torch.diag_embed(sigma_square))
        logprob = normal.log_prob(z) 

        return logprob

    # Compute log-pdf of x under Bernoulli 
    @staticmethod
    def logpdf_bernoulli(x, p):
        # Input:
        #   x: samples [batch_size x data_dimension]
        #   p: the probability of the x being labeled 1 (p is the output of the decoder) [batch_size x data_dimension]
        # Output:
        #   logprob: log-probability of a bernoulli distribution [batch_size]

        # TODO: implement the log likelihood of a bernoulli distribution p(x)  
        logprob = torch.sum((x*torch.log(p) + (1-x)*torch.log(1-p)), dim=1)   

        return logprob
    
    # Sample z ~ q(z|x)
    def sample_z(self, mu, sigma_square):
        # input:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   zs: samples from q(z|x) [batch_size x latent_dimension] 
        zs = self.sample_diagonal_gaussian(mu, sigma_square)

        return zs 


    # Variational Objective
    def elbo_loss(self, sampled_z, mu, sigma_square, x, p):
        # Inputs
        #   sampled_z: samples z from the encoder [batch_size x latent_dimension]
        #   mu:
        #   sigma_square: parameters of q(z|x) [batch_size x latent_dimension]
        #   x: data samples [batch_size x data_dimension]
        #   p: the probability of a pixel being labeled 1 [batch_size x data_dimension]
        # Output
        #   elbo: the ELBO loss (scalar)

        # log_q(z|x) logprobability of z under approximate posterior N(μ,σ)
        log_q = self.logpdf_diagonal_gaussian(sampled_z, mu, sigma_square)
        
        # log_p_z(z) log probability of z under prior
        z_mu    = torch.FloatTensor([0]*self.latent_dimension).repeat(sampled_z.shape[0], 1)
        z_sigma = torch.FloatTensor([1]*self.latent_dimension).repeat(sampled_z.shape[0], 1)
        log_p_z = self.logpdf_diagonal_gaussian(sampled_z, z_mu, z_sigma)

        # log_p(x|z) - conditional probability of data given latents.
        log_p = self.logpdf_bernoulli(x, p)
        
        # TODO: implement the ELBO loss using log_q, log_p_z and log_p
        elbo = torch.mean(log_p + log_p_z - log_q)

        return elbo


    def train(self):
        
        # Set-up ADAM optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        adam_optimizer = optim.Adam(params)

        # Train for ~200 epochs 
        num_batches = int(np.ceil(len(self.train_images) / self.batch_size))
        num_iters = self.num_epoches * num_batches
        
        for i in range(num_iters):
            x_minibatch = self.train_images[batch_indices(i, num_batches, self.batch_size),:]
            adam_optimizer.zero_grad()

            mu, sigma_square = self.encoder(x_minibatch)
            zs = self.sample_z(mu, sigma_square)
            p = self.decoder(zs)
            elbo = self.elbo_loss(zs, mu, sigma_square, x_minibatch, p)
            total_loss = -elbo
            total_loss.backward()
            adam_optimizer.step()

            if i%100 == 0:
                print("Epoch: " + str(i//num_batches) + ", Iter: " + str(i) + ", ELBO:" + str(elbo.item()))

        # Save Optimized Model Parameters
        torch.save(self.encoder.state_dict(), self.e_path)
        torch.save(self.decoder.state_dict(), self.d_path)


    # Generate digits using the VAE
    def visualize_data_space(self):

        with torch.no_grad():

          self.encoder.load_state_dict(torch.load(self.e_path))
          self.decoder.load_state_dict(torch.load(self.d_path))

          pxz_img = np.zeros(shape=(28, 280))
          x_img   = np.zeros(shape=(28, 280))

          pos = np.arange(28)
          for i in range(10):

              # TODO: Sample 10 z from prior 
              z = self.sample_diagonal_gaussian(torch.zeros([1, self.latent_dimension]), torch.ones([1, self.latent_dimension]))

              # TODO: For each z, plot p(x|z)
              pxz = self.decoder(z)
              pxz_img[:,pos] = array_to_image(pxz.numpy())
              figure(figsize=(8, 6), dpi=80)
              plt.imshow(pxz_img)
              plt.show()

              # TODO: Sample x from p(x|z) 
              x = self.sample_Bernoulli(pxz)
              x_img[:,pos] = array_to_image(x.numpy())
              figure(figsize=(8, 6), dpi=80)
              plt.imshow(x_img)
              plt.show()

              pos += 28
        
    # Produce a scatter plot in the latent space, where each point in the plot will be the mean vector 
    # for the distribution $q(z|x)$ given by the encoder. Further, we will colour each point in the plot 
    # by the class label for the input data. Each point in the plot is colored by the class label for 
    # the input data.
    # The latent space should have learned to distinguish between elements from different classes, even though 
    # we never provided class labels to the model!
    def visualize_latent_space(self):
        
        with torch.no_grad():
          # TODO: Encode the training data self.train_images
          mu, sigma_square = self.encoder(self.train_images)

          # TODO: Take the mean vector of each encoding
          means = mu

          # TODO: Plot these mean vectors in the latent space with a scatter
          # Colour each point depending on the class label 
          for i in range(10000):
              ind = torch.argmax(self.train_labels[i,:])
              if ind == 0:
                  plt.scatter(mu[i,0], mu[i,1], color='red')
              elif ind == 1:
                  plt.scatter(mu[i,0], mu[i,1], color='blue')
              elif ind == 2:
                  plt.scatter(mu[i,0], mu[i,1], color='green')
              elif ind == 3:
                  plt.scatter(mu[i,0], mu[i,1], color='orange')
              elif ind == 4:
                  plt.scatter(mu[i,0], mu[i,1], color='purple')
              elif ind == 5:
                  plt.scatter(mu[i,0], mu[i,1], color='brown')
              elif ind == 6:
                  plt.scatter(mu[i,0], mu[i,1], color='cyan')
              elif ind == 7:
                  plt.scatter(mu[i,0], mu[i,1], color='grey')
              elif ind == 8:
                  plt.scatter(mu[i,0], mu[i,1], color='pink')
              elif ind == 9:
                  plt.scatter(mu[i,0], mu[i,1], color='olive')
          plt.show()  
        


    # Function which gives linear interpolation z_α between za and zb
    @staticmethod
    def interpolate_mu(mua, mub, alpha = 0.5):
        return alpha*mua + (1-alpha)*mub


    # A common technique to assess latent representations is to interpolate between two points.
    # Here we will encode 3 pairs of data points with different classes.
    # Then we will linearly interpolate between the mean vectors of their encodings. 
    # We will plot the generative distributions along the linear interpolation.
    def visualize_inter_class_interpolation(self):

        with torch.no_grad():     
            # TODO: Sample 3 pairs of data with different classes
            mu1, sigma_square1 = self.encoder(self.train_images[1,:])
            mu2, sigma_square2 = self.encoder(self.train_images[40,:])
            mu3, sigma_square3 = self.encoder(self.train_images[100,:])

            # TODO: Encode the data in each pair, and take the mean vectors
            interp_mu1_mu2 = self.interpolate_mu(torch.reshape(mu1, (1,2)), torch.reshape(mu2, (1,2)))
            interp_mu2_mu3 = self.interpolate_mu(torch.reshape(mu2, (1,2)), torch.reshape(mu3, (1,2)))
            interp_mu1_mu3 = self.interpolate_mu(torch.reshape(mu1, (1,2)), torch.reshape(mu3, (1,2)))

            interp_ss1_ss2 = self.interpolate_mu(sigma_square1, sigma_square2)
            interp_ss2_ss3 = self.interpolate_mu(sigma_square2, sigma_square3)
            interp_ss1_ss3 = self.interpolate_mu(sigma_square1, sigma_square3)

            # TODO: Linearly interpolate between these mean vectors (Use the function interpolate_mu)
            
            print(mu1.size())
            # TODO: Along the interpolation, plot the distributions p(x|z_α)
            z1 = self.sample_diagonal_gaussian(interp_mu1_mu2, interp_ss1_ss2)
            z2 = self.sample_diagonal_gaussian(interp_mu2_mu3, interp_ss2_ss3)
            z3 = self.sample_diagonal_gaussian(interp_mu1_mu3, interp_ss1_ss3)

            pxz1 = self.decoder(z1)
            pxz2 = self.decoder(z2)
            pxz3 = self.decoder(z3)
            pxz1_img = array_to_image(pxz1.numpy())
            figure(figsize=(8, 6), dpi=80)
            plt.imshow(pxz1_img)
            plt.show()
            pxz2_img = array_to_image(pxz2.numpy())
            figure(figsize=(8, 6), dpi=80)
            plt.imshow(pxz2_img)
            plt.show()
            pxz3_img = array_to_image(pxz3.numpy())
            figure(figsize=(8, 6), dpi=80)
            plt.imshow(pxz3_img)
            plt.show()
        
      

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--e_path', type=str, default="./e_params.pkl", help='Path to the encoder parameters.')
    parser.add_argument('--d_path', type=str, default="./d_params.pkl", help='Path to the decoder parameters.')
    parser.add_argument('--hidden_units', type=int, default=500, help='Number of hidden units of the encoder and decoder models.')
    parser.add_argument('--latent_dimension', type=int, default='2', help='Dimensionality of the latent space.')
    parser.add_argument('--data_dimension', type=int, default='784', help='Dimensionality of the data space.')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--num_epoches', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')

    args = parser.parse_args()
    return args


def main():
    
    # read the function arguments
    args = parse_args()

    # set the random seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # train the model 
    vae = VAE(args)
    vae.train()

    # visualize the latent space
    vae.visualize_data_space()
    vae.visualize_latent_space()
    vae.visualize_inter_class_interpolation()

# Run the main function
main()