import os
import urllib.request
import zipfile


def sample_image_load(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    save_path = os.path.join(data_dir, "hymenoptera_data.zip")

    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

    zip = zipfile.ZipFile(save_path)
    zip.extractall(data_dir)
    zip.close()

    os.remove(save_path)

if __name__ == '__main__':
    data_dir = "data"
    sample_image_load(data_dir)
