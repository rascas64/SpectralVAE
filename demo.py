import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions


class SpectralWrapper(nn.Module):
    """
    Converts a dict of CNNs (one for each continous spectral domain)
    into a single CNN.
    """
    def __init__(self, models):
        super(SpectralWrapper, self).__init__()
        self.models = nn.ModuleDict(models)

    @property
    def out_channels(self):
        with torch.no_grad():
            n_channels = sum([model.n_channels for model in self.models.values()])
            x = torch.ones((2, n_channels))
            x = self.forward(x)
        return x.numel()//2
    
    def forward(self, x, bands):
        z, B = {}, 0

        for model_id, n in zip(self.models.keys(), bands):
            z[model_id] = self.models[model_id](x[:, :, B:B+n])
            B += n

        keys = list(z.keys())
        out = torch.cat([z[keys[i]] for i in range(len(z))], dim=-1)
        out_bands = [z[keys[i]].shape[-1] for i in range(len(z))]

        return out, out_bands


class SequentialSpectralWrapper(nn.Module):
    """
    Converts a dict of CNNs (one for each continous spectral domain)
    into a single CNN.
    """
    def __init__(self, models):
        super(SequentialSpectralWrapper, self).__init__()
        self.models = nn.ModuleDict(models)

    def forward(self, z, x, bands):
        y, B = {}, 0
        model_ids = list(self.models.keys())
        # previous_x = torch.zeros_like(x)[:, :, -self.models[model_ids[0]][0].z_dim.item():]
        previous_x = z[:, -self.models[model_ids[0]][0].z_dim.item():].unsqueeze(1).repeat(1, x.shape[1], 1)
        for model_id, n in zip(self.models.keys(), bands):
            y[model_id] = F.relu(self.models[model_id][0](previous_x, x[:, :, B:B+n]))
            previous_x = x[:, :, B:B+n]
            B += n

        keys = list(y.keys())
        out = torch.cat([y[keys[i]] for i in range(len(y))], dim=-1)
        out_bands = [y[keys[i]].shape[-1] for i in range(len(y))]

        return out, out_bands


class MaskedConvolution(nn.Conv1d):

    def __init__(self, mask_type, z_dim, *args, **kwargs):
        super(MaskedConvolution, self).__init__(*args, **kwargs)

        # Vérifier que le type de masque est autorisé
        assert mask_type in ['A', 'B'], "Type invalide"
        self.mask_type = mask_type

        self.register_buffer('mask', self.weight.data.clone())
        self.register_buffer('z_dim', torch.tensor([z_dim]))

        _, depth, width = self.weight.size()

        # Poids de la convolution :
        # on débute avec des 1 partout (tous les pixels sont considérés)
        self.mask.fill_(1)

        if mask_type == 'A':
            self.mask[:,:, width//2:] = 0
        elif mask_type == 'B':
            self.mask[:,:, width//2 + 1:] = 0
        
        self.n_padding = self.in_channels * self.padding[0]
        self.linear = nn.Linear(self.in_channels * z_dim, self.n_padding)
            
    def compute_padding(self, previous_x):
        padding = torch.zeros((previous_x.shape[0], self.in_channels, self.padding[0] * 2 + (self.padding[0] * 2)%1))
        padding_values = self.linear(previous_x[:, :, -self.z_dim[0]:].reshape(previous_x.shape[0], -1))
        padding_values = padding_values.view(padding_values.shape[0], self.in_channels, -1)
        padding[:, :, :self.padding[0]] = padding_values
        return padding

    def forward(self, previous_x, x):
        # Le filtre convolutif est "masqué" par multiplication avec le masque binaire
        self.weight.data *= self.mask
        padding = self.compute_padding(previous_x)
        x = torch.cat((padding[:, :, :self.padding[0]], x, padding[:, :, self.padding[0]: ]), dim=-1)
        original_padding = self.padding
        self.padding = tuple([0])
        out = super(MaskedConvolution, self).forward(x)
        self.padding = original_padding
        return out


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ClassicVAE(nn.Module):
    def __init__(self, bbl, z_dim=8, channels=8, sigma=1e-3):
        super(ClassicVAE, self).__init__()
        self.continuous_sequences = get_continuous_bands(bbl)
        self.sigma = sigma
        
        conv_params = dict(
            (conv_id, {'kernel_size': n_channels//5, 'out_channels': 16})
            for (conv_id, n_channels) in enumerate(self.continuous_sequences)
        )
        
        # --------------------------- Encoder ------------------------------ #
        
        encoder_convs = {}
        
        self.convs_1 = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.Conv1d(in_channels=1, **params)) 
                 for conv_id, params in conv_params.items())
        )
        
        self.max_pool = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.MaxPool1d(2)) 
                 for conv_id, params in conv_params.items())
        )
        
        self.convs_2 = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.Conv1d(in_channels=params['out_channels'], **params)) 
                 for conv_id, params in conv_params.items())
        )
        
        self.encoder_fc = nn.Linear(16 * 47, 128)
        self.encoder_fc_mu = nn.Linear(128, z_dim)
        self.encoder_fc_logvar = nn.Linear(128, z_dim)
        
        # --------------------------- Decoder ------------------------------ #
        self.decoder_fc = nn.Linear(z_dim, 16 * 47)        
        
        self.tconvs_1 = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.ConvTranspose1d(in_channels=params['out_channels'], **params)) 
                 for conv_id, params in conv_params.items())
        )
        
        self.upsample = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.Upsample(scale_factor=2)) 
                 for conv_id, params in conv_params.items())
        )
        
        padding = [[1, 0], [1, 0], [0, 0], [1, 0], [1, 0], [1, 0]]
        self.padding = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.ReplicationPad1d(pad)) 
                 for conv_id, pad in enumerate(padding))
        )
        
        self.tconvs_2 = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.ConvTranspose1d(in_channels=params['out_channels'], 
                                               out_channels=1, 
                                               kernel_size=params['kernel_size'])) 
                 for conv_id, params in conv_params.items())
        )
        
        self.vae_prior = torch.distributions.normal.Normal(torch.zeros(z_dim), torch.ones(z_dim))
        
    def vae_encoder(self, x):
        x = x.unsqueeze(1)
        x, bands = self.convs_1(x, self.continuous_sequences)
        x = F.relu(x)
        x, bands = self.max_pool(x, bands)
        x = F.relu(x)
        x, bands = self.convs_2(x, bands)
        x = F.relu(x)
        x_shape = x.shape
        x = x.view(x.shape[0], -1)
        z = F.relu(self.encoder_fc(x))
        mu = self.encoder_fc_mu(z)
        logvar = self.encoder_fc_logvar(z)
        z = mu + torch.normal(torch.zeros_like(mu), torch.ones_like(mu)) * torch.exp(logvar / 2)
        return mu, logvar, z, x_shape, bands
        
    def decoder(self, z, x_shape, bands):
        y = F.relu(self.decoder_fc(z))
        y = y.view(x_shape)
        y, bands = self.tconvs_1(y, bands)
        y = F.relu(y)
        y, bands = self.upsample(y, bands)
        y, bands = self.padding(y, bands)
        y, bands = self.tconvs_2(y, bands)
        y = y.squeeze(1)
        return y

    def forward(self, x):
        mu, logvar, z, x_shape, bands = self.vae_encoder(x)
        x = self.decoder(z, x_shape, bands)
        return mu, logvar, z, x
    
    def lossFunc(self, x, y, mu, logvar, beta = 0.1):
        KLD = self.kld(mu, logvar)
        loss = F.mse_loss(y, x) + beta * KLD
        return loss, self.nll(x), KLD, None

    def kld(self, mu, logvar):
        posterior = torch.distributions.normal.Normal(mu, torch.exp(logvar / 2))
        kld_ = torch.distributions.kl.kl_divergence(posterior, self.vae_prior).mean()
        return kld_
    
    def nll(self, x):
        likelihood = torch.distributions.normal.Normal(x, self.sigma * torch.ones_like(x))
        return likelihood.log_prob(x).mean()
    
    def marginal_likelihood(self, loader, num_samples=10):
        marginal_likelihoods = []
        for batch in loader:
            marg_likelihood = torch.zeros((batch.shape[0], num_samples))

            for k in range(num_samples):
                mu, logvar, z, x_shape, bands = self.vae_encoder(batch)
                mu_x = self.decoder(z, x_shape, bands)
                posterior = torch.distributions.normal.Normal(mu, torch.exp(logvar / 2))
                likelihood = torch.distributions.normal.Normal(mu_x, self.sigma * torch.ones_like(mu_x))

                p_z = torch.sum(self.vae_prior.log_prob(z), dim=-1)
                q_z_x = torch.sum(posterior.log_prob(z), dim=-1)
                p_x_z = torch.sum(likelihood.log_prob(batch), dim=-1)

                marg_likelihood[:, k] = p_x_z + p_z - q_z_x
            marginal_likelihoods.extend((torch.logsumexp(marg_likelihood, dim=-1) - np.log(num_samples)).detach().numpy())
        return marginal_likelihoods


class SpectralVAE(nn.Module):
    def __init__(self, bbl, z_dim=8, channels=8):
        super(SpectralVAE, self).__init__()
        self.sigma = 1e-3
        self.continuous_sequences = get_continuous_bands(bbl)
        self.zDim = z_dim
        
        conv_params = dict(
            (conv_id, {'kernel_size': n_channels//5, 'out_channels': 16})
            for (conv_id, n_channels) in enumerate(self.continuous_sequences)
        )
        
        # --------------------------- Encoder ------------------------------ #
        
        encoder_convs = {}
        
        self.convs_1 = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.Conv1d(in_channels=1, **params)) 
                 for conv_id, params in conv_params.items())
        )
        
        self.max_pool = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.MaxPool1d(2)) 
                 for conv_id, params in conv_params.items())
        )
        
        self.convs_2 = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.Conv1d(in_channels=params['out_channels'], **params)) 
                 for conv_id, params in conv_params.items())
        )
        
        self.encoder_fc = nn.Linear(16 * 47, 128)
        self.encoder_fc2 = nn.Linear(128, 2*z_dim)
        
        # --------------------------- Decoder ------------------------------ #
        self.decoder_fc = nn.Linear(z_dim, 16 * 47)
        self.decoder_fc2 = nn.Linear(16 * 47, 186)
        
        """
        self.tconvs_1 = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.ConvTranspose1d(in_channels=params['out_channels'], **params)) 
                 for conv_id, params in conv_params.items())
        )
        
        self.upsample = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.Upsample(scale_factor=2)) 
                 for conv_id, params in conv_params.items())
        )
        
        padding = [[1, 0], [1, 0], [0, 0], [1, 0], [1, 0], [1, 0]]
        self.padding = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.ReplicationPad1d(pad)) 
                 for conv_id, pad in enumerate(padding))
        )
        
        self.tconvs_2 = SpectralWrapper(
            dict((f'conv-{conv_id}', nn.ConvTranspose1d(in_channels=params['out_channels'], 
                                               out_channels=1, 
                                               kernel_size=params['kernel_size'])) 
                 for conv_id, params in conv_params.items())
        )
        """
        # --------------------------- CNN ------------------------------ #
        
        def conv_block(mask_type, in_channels, out_channels, kernel):
            return nn.Sequential(
                MaskedConvolution(mask_type, z_dim, in_channels, out_channels, kernel_size=kernel, stride=1, padding=kernel//2, bias=True),
                nn.ReLU()
        )

        self.conv_block1 = SequentialSpectralWrapper(
            dict((f'conv-{conv_id}', conv_block('A', 2, channels, kernel_size))
                for conv_id, kernel_size in enumerate([23, 9, 9, 11, 19, 23]))
        )
        
        self.conv_block2 = SequentialSpectralWrapper(
            dict((f'conv-{conv_id}', conv_block('B', channels, channels, kernel_size))
                for conv_id, kernel_size in enumerate([15, 5, 9, 9, 15, 15]))
        )
        
        self.conv_block3 = SequentialSpectralWrapper(
            dict((f'conv-{conv_id}', conv_block('B', channels, channels, kernel_size))
                for conv_id, kernel_size in enumerate([9, 3, 5, 5, 7, 9]))
        )

        self.conv_out = nn.Conv1d(channels, 1, kernel_size=1)

        self.prior = torch.distributions.normal.Normal(torch.zeros(self.zDim), torch.ones(self.zDim))

    def kld(self):
        prior = torch.distributions.normal.Normal(
            torch.zeros_like(self.posterior.loc), torch.ones_like(self.posterior.scale)
        )
        kld = torch.distributions.kl.kl_divergence(self.posterior, prior).mean()
        return kld

    def vae_forward(self, x):
        x = x.unsqueeze(1)
        x, bands = self.convs_1(x, self.continuous_sequences)
        x = F.relu(x)
        x, bands = self.max_pool(x, bands)
        x = F.relu(x)
        x, bands = self.convs_2(x, bands)
        x = F.relu(x)
        x_shape = x.shape
        x = x.view(x.shape[0], -1)
        x = F.relu(self.encoder_fc(x))
        x = self.encoder_fc2(x)
        
        mu, logvar = x[:, :self.zDim], x[:, self.zDim:]

        posterior = torch.distributions.normal.Normal(mu, torch.exp(logvar / 2))
        
        z = posterior.rsample()
        y = F.relu(self.decoder_fc(z))
        y = F.relu(self.decoder_fc2(y))
        y = y.unsqueeze(1)
        return mu, logvar, z, y
    
    def cnn_forward(self, z, x):
        x, bands = self.conv_block1(z, x, self.continuous_sequences)
        x = F.relu(x)
        x, bands = self.conv_block2(z, x, bands)
        x = F.relu(x)
        x, bands = self.conv_block3(z, x, bands)
        x = F.relu(x)
        x = torch.sigmoid(self.conv_out(x)).squeeze(1)
        return x
    
    def forward(self, x):
        self.prior = torch.distributions.normal.Normal(torch.zeros(self.zDim), torch.ones(self.zDim))
        mu, logvar, z, y = self.vae_forward(x)
        x = self.cnn_forward(z, torch.cat([x.unsqueeze(1), y], dim=1))
        self.posterior = torch.distributions.normal.Normal(mu, torch.exp(logvar / 2))
        self.likelihood = torch.distributions.normal.Normal(x, self.sigma * torch.ones_like(x))
        return z, y, x

    def lossFunc(self, x, y, beta = None):

        # Distribution Loss
        means = y.mean(dim = 0)
        scale = torch.sqrt(((y-means)**2).sum(dim=0)/(y.shape[0]-1))
        distribution = torch.distributions.Normal(loc = means, scale = scale)
        nll = -distribution.log_prob(x).sum(dim = 1)

        # Kullback-Leiber Divergence
        KLD = self.kld()

        return F.mse_loss(y, x), nll.mean(), KLD, None
            
    def marginal_likelihood(self, loader, num_samples=10):
        marginal_likelihoods = []
        for batch in loader:
            marg_likelihood = torch.zeros((batch.shape[0], num_samples))
            for k in range(num_samples):
                z, _, _= self(batch)

                p_z = torch.sum(self.prior.log_prob(z), dim=-1)
                q_z_x = torch.sum(self.posterior.log_prob(z), dim=-1)
                p_x_z = torch.sum(self.likelihood.log_prob(batch), dim=-1)

                marg_likelihood[:, k] = p_x_z + p_z - q_z_x
            marginal_likelihoods.extend((torch.logsumexp(marg_likelihood, dim=-1) - np.log(num_samples)).detach().numpy())
        return marginal_likelihoods


def get_continuous_bands(bbl):
    n_bands = []
    good_bands = np.where(bbl == True)[0]
    s = 1
    for i in range(len(good_bands)-1):
        if good_bands[i] == good_bands[i+1]-1:
            s += 1
        else:
            n_bands.append(s)
            s = 1

    n_bands.append(s)
    return n_bands


def plotMLIntensityMap(testML, im, saveFolder = None):
    map = testML.reshape(im.shape[0:2])

    fig, ax = plt.subplots(figsize=(15, 10), nrows=1, ncols=2)
    plot = ax[0].imshow(map)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(plot, cax=cax)
    ax[0].set_title("Valeurs de vraisemblance marginale dans l'image")
    plot = ax[1].imshow(im[:, :, 100])
    ax[1].set_title("Bande n°100/186")
    fig.suptitle("Comparaison entre les valeurs de vraisemblance marginale obtenues et l'image originale")
    if saveFolder is not None:
        plt.savefig(saveFolder+"/ML intensity map.pdf")
        plt.close()


if __name__ == "__main__":
    # Data 
    spectra = "data/spectra.npy"
    wv = "data/wv.npy"
    factors = "data/factors.npy"
    bbl = "data/bbl.npy"

    spectra = np.load(spectra)
    factors = np.load(factors)
    labels = factors[:, -1]
    wv = np.load(wv)
    bbl = np.load(bbl)
    
    im = np.load("data/train_img.npy")
    testLabels = np.load("data/train_img_labels.npy")
    tmp = np.concatenate((testLabels.reshape(im.shape[0], im.shape[1], 1), im), axis=2)

    spectra = np.hstack((labels.reshape(-1, 1), spectra))
    spectra = spectra.astype(np.float32)

    testData = torch.from_numpy(tmp).view(-1, tmp.shape[-1]).float()
    testData = testData[:, np.insert(bbl, 0, True)]

    import pdb
    pdb.set_trace()

    # Models
    vae = ClassicVAE(bbl)
    spectralvae = SpectralVAE(bbl)

    # Loading state dicts 
    vae.load_state_dict(torch.load("pretrained nets/ClassicVAE.pth"))
    spectralvae.load_state_dict(torch.load("pretrained nets/SpectralVAE.pth"))    

    # Marginal Log Likelihood intensity map
    testML = vae.marginal_likelihood(torch.utils.data.DataLoader(testData[:, 1:], batch_size=64, shuffle=False))
    testML = np.clip(testML, -20000, 3000)
    plotMLIntensityMap(testML, im[:, :, bbl])

    testML = spectralvae.marginal_likelihood(torch.utils.data.DataLoader(testData[:, 1:], batch_size=64, shuffle=False))
    testML = np.clip(testML, -20000, 3000)
    plotMLIntensityMap(testML, im[:, :, bbl])

    # Histograms
    testData = torch.from_numpy(tmp).view(-1, tmp.shape[-1]).float()
    testData = testData[:, np.insert(bbl, 0, True)]
    testML = net1.marginal_likelihood(torch.utils.data.DataLoader(testData[:, 1:], batch_size=64, shuffle=False))
    testML = np.clip(testML, -20000, 3000)
    plotMLIntensityMap(testML, im[:, :, bbl])
    trainData = data.trainData.dataset[data.trainData.indices, :]
    valData = data.valData.dataset[data.valData.indices, :]

    classLabels =  {0: "Végétation", 1: "Tuile", 2: "Terre sableuse", 3: "Asphalte", 4: "Tôle"}

    testData = np.vstack((valData, testData))
    trainLabels, testLabels = trainData[:, 0], testData[:, 0]
    train, test = torch.from_numpy(trainData[:, 1:]), torch.from_numpy(testData[:, 1:])

    trainML = vae.marginal_likelihood(torch.utils.data.DataLoader(train, batch_size=64, shuffle=False))
    testML = vae.marginal_likelihood(torch.utils.data.DataLoader(test, batch_size=64, shuffle=False))

    trainML = np.clip(trainML, -20000, 3000)
    testML = np.clip(testML, -20000, 3000)

    plotHistograms(trainML, trainLabels, testML, testLabels, classLabels)

    plt.show()