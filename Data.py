
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from tqdm import tqdm
import os 


def kaggle_dataset(args):

    if args.dataset == "pokemon":
        path = 'lantian773030/pokemonclassification'
        out_path = "dataset"
    
    elif args.dataset == "anime":
        path = "arnaud58/ffhq-flickr-faces-align-crop-and-segment"
        try:
            os.mkdir('Images')
        except:
            pass

        out_path = "dataset/Images"
    file_name = path.split('/')[-1] + ".zip"

    while not args.kaggle_user:
        print("\n REQUIRED KAGGLE API USERNAME TO DOWNLOAD KAGGLE DATASET\n")
        args.kaggle_user = input("KAGGLE USERNAME FROM KAGGLE.JSON")

    while not args.kaggle_key:
        print("\n REQUIRED KAGGLE API KEY TO DOWNLOAD KAGGLE DATASET\n")
        args.kaggle_user = input("KAGGLE API KEY FROM KAGGLE.JSON")
    
    os.environ['KAGGLE_USERNAME'] = args.kaggle_user
    os.environ['KAGGLE_KEY'] = args.kaggle_key
    
        
    print("###### Authenticating Kaggle API Dataset #####")
    api = KaggleApi()
    api.authenticate()

    # "Better way to check and re download save size and structre in a file and check if full file is there"
    try:
        os.remove(file_name)
    except:
        pass

    print("######Downloading Dataset #####")
    api.dataset_download_files(path, quiet=False)

    print("#######Extracting Dataset #####")

    with ZipFile(file=file_name) as zip_file:
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            zip_file.extract(member=file, path=out_path)

    print("#######Deleting Zip File #####")
    os.remove(file_name)

def MNIST_downlaod(args):
    pass


if __name__ == "__main__":
    pass