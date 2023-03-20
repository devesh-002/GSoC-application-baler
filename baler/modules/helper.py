import argparse
import os
import pickle
import sys

import numpy
import pandas
import torch

from modules import training, plotting, data_processing
from dataclasses import dataclass
import importlib

def get_arguments():
    parser = argparse.ArgumentParser(
        prog="poetry run baler",
        description="Baler is a machine learning based compression tool for scientific data.",
    )
    parser.add_argument(
        "--project", type=str, required=False, help="Name of new project, it is a directory name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        help="new_project, preprocessing (pp), train, compress, decompress, info, evaluate (eval), analysis",
    )

    args = parser.parse_args()
    if not args.mode or (args.mode != "new_project" and not args.project):
        parser.print_usage()
        exit(1)
    if args.mode == "new_project":
        config = None
    else:
        config = configClass
        importlib.import_module(f"projects.{args.project}.{args.project}_config").set_config(config)
    return config, args.mode, args.project


def create_new_project(project_name: str, base_path: str = "projects") -> None:
    project_path = os.path.join(base_path, project_name)
    if os.path.exists(project_path):
        print(f"The project {project_path} already exists.")
        return

    required_directories = [
        "compressed_output",
        "decompressed_output",
        "plotting",
        "training",
        "model",
    ]
    os.makedirs(project_path)
    with open(os.path.join(project_path, f"{project_name}_config.py"), "w") as f:
        f.write(create_default_config(project_name))
    with open(os.path.join(project_path, f"{project_name}_preprocessing.py"), "w") as f:
        f.write(create_default_preprocessing(project_name))
    with open(os.path.join(project_path, f"{project_name}_analysis.py"), "w") as f:
        f.write(create_default_analysis(project_name))
    for directory in required_directories:
        os.makedirs(os.path.join(project_path, directory))


@dataclass
class configClass:
    input_path          : str
    compression_ratio   : float
    epochs              : int
    early_stopping      : bool
    lr_scheduler        : bool
    patience            : int
    min_delta           : int
    model_name          : str
    custom_norm         : bool
    l1                  : bool
    reg_param           : float
    RHO                 : float
    lr                  : float
    batch_size          : int
    save_as_root        : bool
    test_size           : float

def create_default_config(project_name) -> str:
    return f'''
def set_config(c):
    c.input_path          = "data/{project_name}/{project_name}.pickle"
    c.path_before_pre_processing = "data/example/example.root"
    c.compression_ratio   = 2.0
    c.epochs              = 10
    c.early_stopping      = False
    c.lr_scheduler        = True
    c.patience            = 20
    c.min_delta           = 0
    c.model_name          = "george_SAE"
    c.custom_norm         = False
    c.l1                  = True
    c.reg_param             = 0.001
    c.RHO                 = 0.05
    c.lr                  = 0.001
    c.batch_size          = 512
    c.save_as_root        = True
    c.test_size           = 0.15
'''

def create_default_preprocessing(project_name) -> str:
    return f'''
import uproot

def pre_processing(input_path,output_path):

    Branch = "Events"
    Collection = "recoGenJets_slimmedGenJets__PAT."
    Objects = "recoGenJets_slimmedGenJets__PAT.obj"
    dropped_variables = [
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fX",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fY",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fZ",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.qx3_",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.pdgId_",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.status_",
        "recoGenJets_slimmedGenJets__PAT.obj.mJetArea",
        "recoGenJets_slimmedGenJets__PAT.obj.mPileupEnergy",
        "recoGenJets_slimmedGenJets__PAT.obj.mPassNumber"]
    
    # Load data
    tree = uproot.open(input_path)[Branch][Collection][Objects]
    #Type clearing
    names = type_clearing(tree)
    df = tree.arrays(names, library="pd")
    # Clean data
    df = df.drop(columns=dropped_variables)
    df = df.reset_index(drop=True)
    df = df.dropna()
    global cleared_column_names
    cleared_column_names = list(df)
    df.to_pickle(output_path)

def type_clearing(tt_tree):
    type_names = tt_tree.typenames()
    column_type = []
    column_names = []

    # In order to remove non integers or -floats in the TTree,
    # we separate the values and keys
    for keys in type_names:
        column_type.append(type_names[keys])
        column_names.append(keys)

    # Checks each value of the typename values to see if it isn't an int or
    # float, and then removes it
    for i in range(len(column_type)):
        if column_type[i] != "float[]" and column_type[i] != "int32_t[]":
            # print('Index ',i,' was of type ',Typename_list_values[i],'\
            # and was deleted from the file')
            del column_names[i]

    # Returns list of column names to use in load_data function
    return column_names
    '''

def create_default_analysis(project_name) -> str:
    return '''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize as opt;
import pickle

#data = "http://opendata.cern.ch/record/21856#"

def analysis(project_name, data_path_before, data_path_after):
    print(data_path_before, data_path_after)
    project_path = f"projects/{project_name}/"

    with open(data_path_before, "rb") as handle:
        before = pickle.load(handle)
        before = before
    with open(data_path_after, "rb") as handle:
        after = pickle.load(handle)

    #plot_all(project_path, before, after)
    variable = "recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fM"
    plot_peak(project_path, before[variable], after[variable])

def fit(x, a, b, c, k, m):
     return a*np.exp(-((x-b)**2)/(2*c**2)) + m*x + k

def plot_peak(project_path, before, after):
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8*2.5*(1/2.54), 6*2.5*(1/2.54)), sharex=True)

    x_min = min(before+after)
    x_max = max(before+after)
    x_diff = abs(x_max - x_min)

    with PdfPages(project_path + "/plotting/analysis.pdf") as pdf:    

        # Before Histogram
        counts_before, bins_before = np.histogram(
            before, bins=np.linspace(100,225,200)
        )
        hist1 = ax1.hist(
            bins_before[:-1], bins_before, weights=counts_before, label="Before", histtype="step", color="black"
        )
        before_bin_centers = hist1[1][:-1]+(hist1[1][1:]-hist1[1][:-1])/2
        before_bin_centers_error = (hist1[1][1:]-hist1[1][:-1])/2
        before_bin_counts = hist1[0]
        before_count_error = np.sqrt(hist1[0])
        ax1.errorbar(before_bin_centers,before_bin_counts, yerr=before_count_error, xerr=None, marker="",linewidth=0.75, markersize=1, linestyle='', color="black")
        optimizedParameters1, pcov1 = opt.curve_fit(fit, before_bin_centers, before_bin_counts, p0=[1,173,1,1,1]);
        perr1 = np.sqrt(np.diag(pcov1))
        ax1.plot(before_bin_centers, fit(before_bin_centers, *optimizedParameters1),linewidth=1, label="Fit",color="red")
        leg1 = ax1.legend(borderpad=0.5, loc=1, ncol=2, frameon=True,facecolor="white",framealpha=1,fontsize='medium')
        leg1._legend_box.align = "left"
        leg1.set_title(f"Mass  : {round(optimizedParameters1[1],2)} +/- {round(perr1[1],2)}" + f"Width : {round(optimizedParameters1[2],2)} +/- {round(perr1[2],2)}")
        ax1.set_ylabel("Counts", fontsize=14, ha='right', y=1.0)
        ax1.set_xlabel("Mass [GeV]", fontsize=14, ha='right',x=1.0)
        ax1.set_title('Before Compression')
        ax1.set_ylim(0,1500)
        print(f"Before compression:")
        print(f"Mass  : {round(optimizedParameters1[1],2)} +/- {round(perr1[1],2)}")
        print(f"Width : {round(optimizedParameters1[2],2)} +/- {round(perr1[2],2)}")

        # After Histogram
        counts_after, bins_after = np.histogram(
            after, bins=np.linspace(100,225,200)
        )
        hist2 = ax2.hist(
            bins_after[:-1], bins_after, weights=counts_after, label="After", histtype="step", color="black"
        )
        after_bin_centers = hist2[1][:-1]+(hist1[1][1:]-hist1[1][:-1])/2
        after_bin_centers_error = (hist2[1][1:]-hist1[1][:-1])/2
        after_bin_counts = hist2[0]
        after_count_error = np.sqrt(hist2[0])
        ax2.errorbar(after_bin_centers,after_bin_counts, yerr=after_count_error, xerr=None, marker="",linewidth=0.75, markersize=1, linestyle='', color="black")
        optimizedParameters2, pcov2 = opt.curve_fit(fit, after_bin_centers, after_bin_counts, p0=[1,173,1,1,1]);
        perr2 = np.sqrt(np.diag(pcov2))
        ax2.plot(after_bin_centers, fit(after_bin_centers, *optimizedParameters2),linewidth=1, label="Fit",color="red");
        leg2 = ax2.legend(borderpad=0.5, loc=1, ncol=2, frameon=True,facecolor="white",framealpha=1,fontsize='medium')
        leg2._legend_box.align = "left"
        leg2.set_title(f"Mass  : {round(optimizedParameters2[1],2)} +/- {round(perr2[1],2)}" + f"Width : {round(optimizedParameters2[2],2)} +/- {round(perr2[2],2)}")
        ax2.set_ylabel("Counts", fontsize=14, ha='right', y=1.0)
        ax2.set_xlabel("Mass [GeV]", fontsize=14, ha='right',x=1.0)
        ax2.set_title('After Decompression')
        ax2.set_ylim(0,1500)

        print(f"After compression:")
        print(f"Mass  : {round(optimizedParameters2[1],2)} +/- {round(perr2[1],2)}")
        print(f"Width : {round(optimizedParameters2[2],2)} +/- {round(perr2[2],2)}")

        diff = round((abs(optimizedParameters1[1]-optimizedParameters2[1])/optimizedParameters2[1])*100,1)
        fig.suptitle(f"Relative Mass Difference = {diff} %", fontsize=16)

        pdf.savefig()
    '''

def to_pickle(data, path):
    with open(path, "wb") as handle:
        pickle.dump(data, handle)

def from_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def model_init(model_name):
    # This is used when we don't have saved model parameters.
    ModelObject = data_processing.initialise_model(model_name)
    return ModelObject


def data_loader(data_path):
    return data_processing.load_data(data_path)


def numpy_to_tensor(data):
    if isinstance(data, pandas.DataFrame):
        data = data.to_numpy()

    return torch.from_numpy(data)


def normalize(data, custom_norm, cleared_col_names):
    data = numpy.apply_along_axis(
        data_processing.normalize, axis=0, arr=data, custom_norm=custom_norm
    )
    df = data_processing.numpy_to_df(data, cleared_col_names)
    return df


def process(data_path,test_size):
    df = data_processing.load_data(data_path)
    cleared_col_names = data_processing.get_columns(df)
    normalization_features = data_processing.find_minmax(df)
    number_of_columns = len(data_processing.get_columns(df))

    train_set, test_set = data_processing.split(
        df, test_size=test_size, random_state=1
    )
    return train_set, test_set, number_of_columns, normalization_features, cleared_col_names


def renormalize(data, true_min_list, feature_range_list):
    return data_processing.renormalize_func(data, true_min_list, feature_range_list)


def train(model, number_of_columns, train_set, test_set, project_path, config):
    if(config.model_name=="adversial"):
        return (training.adversial_training(train_set,test_set,project_path,config))
    return training.train(
        model, number_of_columns, train_set, test_set, project_path, config
    )


def plot(project_path):
    plotting.plot(project_path)


def loss_plotter(path_to_loss_data, output_path, config):
    return plotting.loss_plot(path_to_loss_data, output_path, config)


def model_saver(model, model_path):
    return data_processing.save_model(model, model_path)


def detach(tensor):
    return tensor.cpu().detach().numpy()


def compress(model_path, config):
    # Give the encoding function the correct input as tensor
    data = data_loader(config.input_path)
    cleared_col_names = data_processing.get_columns(data)
    number_of_columns = len(data_processing.get_columns(data))
    try:
        config.latent_space_size = int(number_of_columns//config.compression_ratio)
        config.number_of_columns = number_of_columns
    except AttributeError:
        assert(number_of_columns==config.number_of_columns)
    data_before = numpy.array(data)
    data = normalize(data, config.custom_norm, cleared_col_names)
    
    # Initialise and load the model correctly.
    model_name=config.model_name
    if(model_name=="adversial"):
        model_name="Encoder"

    ModelObject = data_processing.initialise_model(model_name)
    model = data_processing.load_model(
        ModelObject,
        model_path=model_path,
        n_features=number_of_columns,
        z_dim=config.latent_space_size,
    )
    data_tensor = numpy_to_tensor(data).to(model.device)
    if(model_name=="Encoder"):
        compressed=model(data_tensor)
    else:
        compressed = model.encode(data_tensor)
    return compressed, data_before, cleared_col_names


def decompress(model_path, input_path, model_name):
    if(model_name=="adversial"):
        model_name="Decoder"

    # Load the data & convert to tensor
    data = data_loader(input_path)
    latent_space_size = len(data[0])
    modelDict = torch.load(str(model_path))
    number_of_columns = len(modelDict[list(modelDict.keys())[-1]])


    # Initialise and load the model correctly.
    ModelObject = data_processing.initialise_model(model_name)
    model = data_processing.load_model(
        ModelObject,
        model_path=model_path,
        n_features=number_of_columns,
        z_dim=latent_space_size,
    )
    data_tensor = numpy_to_tensor(data).to(model.device)

    if(model_name=="Decoder"):
        decompressed=model(data_tensor)
    else:
        decompressed = model.decode(data_tensor)

    return decompressed


def to_root(data_path, cleared_col_names, save_path):
    if isinstance(data_path, pickle.Pickler):
        df, Names = data_processing.pickle_to_df(file_path=data_path)
        return data_processing.df_to_root(df, Names, save_path)
    elif isinstance(data_path, pandas.DataFrame):
        return data_processing.df_to_root(
            data_path, save_path=save_path
        )
    elif isinstance(data_path, numpy.ndarray):
        df = data_processing.numpy_to_df(data_path, cleared_col_names)
        df_names = df.columns
        return data_processing.df_to_root(df, col_names=df_names, save_path=save_path)


def get_device():
    device = None
    if torch.cuda.is_available():
        dev = "cuda:0"
        device = torch.device(dev)
    else:
        dev = "cpu"
        device = torch.device(dev)
    print(device)
    return device
