import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import modules.utils as utils
from modules import models
import random
import numpy as np
from torch.optim import lr_scheduler
from torch.autograd import Variable

from pytorch_lightning import Trainer
import modules.helper as helper

# import torch.tensor as Tensor



def fit(model, train_dl, train_ds, model_children, regular_param, optimizer, RHO, l1):
    print("### Beginning Training")
    config, mode, project_name = helper.get_arguments()

    model.train()

    running_loss = 0.0
    counter = 0
    n_data = int(len(train_ds) / train_dl.batch_size)
    for inputs, labels in tqdm(
        train_dl, total=n_data, desc="# Training", file=sys.stdout
    ):
        counter += 1
        inputs = inputs.to(model.device)
        optimizer.zero_grad()
        reconstructions ,_= model(inputs)
        loss, mse_loss, l1_loss = utils.sparse_loss_function_L1(
            model_children=model_children,
            true_data=inputs,
            reconstructed_data=reconstructions,
            # encode=encode,
            reg_param=regular_param,
            validate=False,
            # z_dim=config.latent_space_size
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f"# Finished. Training Loss: {loss:.6f}")
    return epoch_loss, mse_loss, l1_loss


def validate(model, test_dl, test_ds, model_children, reg_param):
    print("### Beginning Validating")
    config, mode, project_name = helper.get_arguments()

    model.eval()
    counter = 0
    running_loss = 0.0
    n_data = int(len(test_ds) / test_dl.batch_size)
    with torch.no_grad():
        for inputs, labels in tqdm(
            test_dl, total=n_data, desc="# Validating", file=sys.stdout
        ):
            counter += 1
            inputs = inputs.to(model.device)
            reconstructions,_ = model(inputs)
            # print(reconstructions)
            loss= utils.sparse_loss_function_L1(
                model_children=model_children,
                true_data=inputs,
                reconstructed_data=reconstructions,
                # encode=encode,
                reg_param=reg_param,
                validate=True,
                # z_dim=config.latent_space_size
            )
            running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f"# Finished. Validation Loss: {loss:.6f}")
    return epoch_loss

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(model, variables, train_data, test_data, parent_path, config):

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    # torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(0)

    learning_rate = config.lr
    bs = config.batch_size
    reg_param = config.reg_param
    RHO = config.RHO
    l1 = config.l1
    epochs = config.epochs
    latent_space_size = config.latent_space_size

    model_children = list(model.children())

    # Constructs a tensor object of the data and wraps them in a TensorDataset object.
    train_ds = TensorDataset(
        torch.tensor(train_data.values, dtype=torch.float64),
        torch.tensor(train_data.values, dtype=torch.float64),
    )
    valid_ds = TensorDataset(
        torch.tensor(test_data.values, dtype=torch.float64),
        torch.tensor(test_data.values, dtype=torch.float64),
    )

    # Converts the TensorDataset into a DataLoader object and combines into one DataLoaders object (a basic wrapper
    # around several DataLoader objects).
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False, worker_init_fn=seed_worker, generator=g)
    valid_dl = DataLoader(valid_ds, batch_size=bs, worker_init_fn=seed_worker, generator=g)  ## Used to be batch_size = bs * 2

    ## Select Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ## Activate early stopping
    if config.early_stopping == True:
        early_stopping = utils.EarlyStopping(
            patience=config.patience, min_delta=config.min_delta
        )  # Changes to patience & min_delta can be made in configs

    ## Activate LR Scheduler
    if config.lr_scheduler == True:
        lr_scheduler = utils.LRScheduler(
            optimizer=optimizer, patience=config.patience
        )

    # train and validate the autoencoder neural network
    train_loss = []
    val_loss = []
    start = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        train_epoch_loss, mse_loss_fit, regularizer_loss_fit = fit(
            model=model,
            train_dl=train_dl,
            train_ds=train_ds,
            model_children=model_children,
            optimizer=optimizer,
            RHO=RHO,
            regular_param=reg_param,
            l1=l1,
        )

        train_loss.append(train_epoch_loss)

        val_epoch_loss = validate(
            model=model,
            test_dl=valid_dl,
            test_ds=valid_ds,
            model_children=model_children,
            reg_param=reg_param,
        )
        val_loss.append(val_epoch_loss)
        if config.lr_scheduler:
            lr_scheduler(val_epoch_loss)
        if config.early_stopping:
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                break

    end = time.time()

    print(f"{(end - start) / 60:.3} minutes")
    pd.DataFrame({"Train Loss": train_loss, "Val Loss": val_loss}).to_csv(
        parent_path + "loss_data.csv"
    )

    data_as_tensor = torch.tensor(test_data.values, dtype=torch.float64)
    data_as_tensor = data_as_tensor.to(model.device)
    pred_as_tensor = model(data_as_tensor)

    return data_as_tensor, pred_as_tensor

# special training loop for adversial autoencoder.

def adversial_training( train_data, test_data, parent_path, config):
    epsilon=1e-8
    device = helper.get_device()
## initialise this
    encoder=models.Encoder(
                device=device, n_features=config.number_of_columns, z_dim=config.latent_space_size

    )
    decoder=models.Decoder(
                device=device, n_features=config.number_of_columns, z_dim=config.latent_space_size

    )
    discriminator=models.Discriminator(
                        device=device, n_features=config.number_of_columns, z_dim=config.latent_space_size

    )
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    # torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(0)

    learning_rate = config.lr
    bs = config.batch_size
    reg_param = config.reg_param
    RHO = config.RHO
    l1 = config.l1
    epochs = config.epochs
    latent_space_size = config.latent_space_size

    # model_children = list(model.children())

    # Constructs a tensor object of the data and wraps them in a TensorDataset object.
    train_ds = TensorDataset(
        torch.tensor(train_data.values, dtype=torch.float64),
        torch.tensor(train_data.values, dtype=torch.float64),
    )
    valid_ds = TensorDataset(
        torch.tensor(test_data.values, dtype=torch.float64),
        torch.tensor(test_data.values, dtype=torch.float64),
    )

    # Converts the TensorDataset into a DataLoader object and combines into one DataLoaders object (a basic wrapper
    # around several DataLoader objects).
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False, worker_init_fn=seed_worker, generator=g)
    valid_dl = DataLoader(valid_ds, batch_size=bs, worker_init_fn=seed_worker, generator=g)  ## Used to be batch_size = bs * 2
    
    def discriminator_criterion(input, target, epsilon):
        # print
        x= -torch.mean(torch.log(target + epsilon) + torch.log(1 - input + epsilon))
        # print(x.item())
        return x
    def generator_criterion(input, epsilon):
        return -torch.mean(torch.log(input + epsilon))

    reconstruct_loss_fn=nn.MSELoss()
    optim_encoder=optim.Adam(encoder.parameters(),lr=learning_rate)
    optim_decoder=optim.Adam(decoder.parameters(),lr=learning_rate)
    
    optim_discriminator=optim.Adam(discriminator.parameters(),lr=learning_rate)
    optim_encoder_generator=optim.Adam(encoder.parameters(),lr=learning_rate/10)

    encoder_scheduler = lr_scheduler.StepLR(optim_encoder, step_size=7, gamma=0.1)
    decoder_scheduler = lr_scheduler.StepLR(optim_decoder, step_size=7, gamma=0.1)
    discriminator_scheduler = lr_scheduler.StepLR(optim_discriminator, step_size=7, gamma=0.1)
    generator_scheduler = lr_scheduler.StepLR(optim_encoder_generator, step_size=7, gamma=0.1)

    n_data = int(len(train_ds) / train_dl.batch_size)

    for epoch in range(epochs):
        encoder.train()
        discriminator.train()
        decoder.train()

        en_loss=0.0;de_loss=0.0;discrimin_loss=0.0;counter=0.0

        for inputs, labels in tqdm(
            train_dl, total=n_data, desc="# Training", file=sys.stdout
        ):
            counter+=1
            optim_encoder_generator.zero_grad()
            optim_discriminator.zero_grad()
            optim_decoder.zero_grad()
            optim_encoder.zero_grad()

            # pass through normally

            encode_data=encoder(inputs)
            decode_data=decoder(encode_data)

#  grad descent
            mse_loss=reconstruct_loss_fn(input=decode_data,target=inputs)

            mse_loss.backward()
            optim_encoder.step()
            optim_decoder.step()

# close the encoder now

            encoder.eval()

            z_real = Variable(torch.randn(inputs.size(0), latent_space_size,dtype=torch.float64)).to(device)
            z_fake=encoder(inputs)

            d_real=discriminator(z_real)
            d_fake=discriminator(z_fake)

            discrim_loss=discriminator_criterion(input=d_fake,target=d_real,epsilon=epsilon)
            discrim_loss.backward()
            optim_discriminator.step()
# activate the encoder
            encoder.train()

            z_fake=encoder(inputs)
            d_fake=discriminator(z_fake)

            generator_loss=generator_criterion(input=d_fake,epsilon=epsilon)

            generator_loss.backward()
            optim_encoder_generator.step()

            en_loss+=mse_loss.item()
            de_loss+=discrim_loss.item()
            discrimin_loss+=generator_loss.item()

        reconst_loss_epoch=en_loss/counter
        discrim_loss_epoch=discrimin_loss/counter
        de_loss_epoch=de_loss/counter
        print(f'\n[Epoch {epoch+1}/{epochs}]', 'reconstruction loss: {:.4f}; discriminator loss: {:.4f}; generator loss: {:.4f}'.format(reconst_loss_epoch, discrim_loss_epoch, de_loss_epoch))

        encoder_scheduler.step()
        decoder_scheduler.step()
        discriminator_scheduler.step()
        generator_scheduler.step()
    # end = time.time()
    data_as_tensor = torch.tensor(test_data.values, dtype=torch.float64)
    decoder.to(device)
    discriminator.to(device)
    data_as_tensor = data_as_tensor.to(encoder.device)
    pred_as_tensor = decoder(encoder(data_as_tensor))

    return encoder,decoder, discriminator,data_as_tensor, pred_as_tensor

    

    # return data_as_tensor, pred_as_tensor


    
    # exit(0)