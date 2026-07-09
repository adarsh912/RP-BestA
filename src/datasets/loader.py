import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd

def load_local_ucr_txt(file_path):
    """
    Loads a UCR archive dataset from a space/tab-separated txt file.
    The first column is the label, and the remaining columns are the time series values.
    """
    try:
        data = np.loadtxt(file_path)
        y = data[:, 0]
        X = data[:, 1:]
        return X, y
    except Exception as e:
        # Fallback to pandas for handling irregular spacing or headers
        df = pd.read_csv(file_path, header=None, sep=None, engine='python')
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        return X, y

def load_ts_file(file_path):
    """
    Parses a standard .ts file (used by sktime/aeon) where data is formatted as:
    value1,value2,...,valueN:label
    and contains metadata headers starting with @.
    """
    X = []
    y = []
    in_data = False
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if in_data:
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                vals_str = parts[0].split(',')
                vals = [float(v) for v in vals_str]
                label = parts[1].strip()
                X.append(vals)
                try:
                    if '.' in label:
                        y.append(float(label))
                    else:
                        y.append(int(label))
                except ValueError:
                    y.append(label)
    return np.array(X), np.array(y)

def load_ucr_dataset(dataset_name, data_dir="data"):
    """
    Loads a UCR dataset. Checks local directory, falls back to pyts built-in,
    or fetches from OpenML / public repositories.
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    train_path_txt = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.txt")
    test_path_txt = os.path.join(dataset_dir, f"{dataset_name}_TEST.txt")
    train_path_tsv = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.tsv")
    test_path_tsv = os.path.join(dataset_dir, f"{dataset_name}_TEST.tsv")
    train_path_ts = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.ts")
    test_path_ts = os.path.join(dataset_dir, f"{dataset_name}_TEST.ts")
    
    # 1. Check if files exist locally in txt, tsv, or ts format
    if os.path.exists(train_path_txt) and os.path.exists(test_path_txt):
        X_train, y_train = load_local_ucr_txt(train_path_txt)
        X_test, y_test = load_local_ucr_txt(test_path_txt)
        return X_train, y_train, X_test, y_test
    
    if os.path.exists(train_path_tsv) and os.path.exists(test_path_tsv):
        X_train, y_train = load_local_ucr_txt(train_path_tsv)
        X_test, y_test = load_local_ucr_txt(test_path_tsv)
        return X_train, y_train, X_test, y_test

    if os.path.exists(train_path_ts) and os.path.exists(test_path_ts):
        X_train, y_train = load_ts_file(train_path_ts)
        X_test, y_test = load_ts_file(test_path_ts)
        return X_train, y_train, X_test, y_test

    # 2. Check if the dataset is built into pyts.datasets to avoid network requests
    try:
        if dataset_name.lower() == "gunpoint":
            from pyts.datasets import load_gunpoint
            X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
            return X_train, y_train, X_test, y_test
        elif dataset_name.lower() == "coffee":
            from pyts.datasets import load_coffee
            X_train, X_test, y_train, y_test = load_coffee(return_X_y=True)
            return X_train, y_train, X_test, y_test
    except ImportError:
        pass  # pyts is not installed yet or loading failed, try next
    # 3. Try to fetch from the official UCR aeon-formatted zip archive
    try:
        print(f"Attempting to download '{dataset_name}' from the official UCR archive...")
        zip_url = f"https://timeseriesclassification.com/aeon-formatted/{dataset_name}.zip"
        zip_path = os.path.join(dataset_dir, f"{dataset_name}.zip")
        urllib.request.urlretrieve(zip_url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
            
        os.remove(zip_path)
        
        if os.path.exists(train_path_ts) and os.path.exists(test_path_ts):
            X_train, y_train = load_ts_file(train_path_ts)
            X_test, y_test = load_ts_file(test_path_ts)
            return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"Official UCR archive download failed: {e}")
        # Clean up zip on failure
        zip_path_temp = os.path.join(dataset_dir, f"{dataset_name}.zip")
        if os.path.exists(zip_path_temp):
            os.remove(zip_path_temp)

    # 4. Try to fetch from OpenML (highly reliable repository mirror)
    try:
        from sklearn.datasets import fetch_openml
        print(f"Attempting to fetch dataset '{dataset_name}' from OpenML...")
        data = fetch_openml(name=dataset_name, version=1, as_frame=False)
        X = data.data
        y = data.target
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        np.savetxt(train_path_txt, np.column_stack((y_train.astype(float), X_train)))
        np.savetxt(test_path_txt, np.column_stack((y_test.astype(float), X_test)))
        
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"OpenML fetch failed: {e}")
        
    # 4. Try loading from other public GitHub mirrors
    # Many common UCR datasets are mirrored in these raw repositories:
    mirrors = [
        # sktime mirror contains .ts files
        (f"https://raw.githubusercontent.com/sktime/sktime/main/sktime/datasets/data/{dataset_name}/{dataset_name}_TRAIN.ts", train_path_ts, ".ts"),
        # cd-diagram mirror contains .tsv files
        (f"https://raw.githubusercontent.com/hfawaz/cd-diagram/master/{dataset_name}/{dataset_name}_TRAIN.tsv", train_path_tsv, ".tsv"),
        # anomaly mirror contains txt files
        (f"https://raw.githubusercontent.com/ajbagwell/UCR-Time-Series-Archive-2015/master/UCR%20Time%20Series%20Anomaly%20Archive/{dataset_name}/{dataset_name}_TRAIN", train_path_txt, ".txt"),
    ]
    
    for mirror_url, local_path, extension in mirrors:
        try:
            print(f"Trying to fetch train data from mirror: {mirror_url}")
            urllib.request.urlretrieve(mirror_url, local_path)
            
            test_mirror_url = mirror_url.replace("TRAIN", "TEST")
            local_test_path = local_path.replace("TRAIN", "TEST")
            
            print(f"Trying to fetch test data from mirror: {test_mirror_url}")
            urllib.request.urlretrieve(test_mirror_url, local_test_path)
            
            if extension == ".ts":
                X_train, y_train = load_ts_file(local_path)
                X_test, y_test = load_ts_file(local_test_path)
            else:
                X_train, y_train = load_local_ucr_txt(local_path)
                X_test, y_test = load_local_ucr_txt(local_test_path)
            return X_train, y_train, X_test, y_test
        except Exception as e:
            # Clean up on failure
            if os.path.exists(local_path):
                os.remove(local_path)
            test_path_temp = local_path.replace("TRAIN", "TEST")
            if os.path.exists(test_path_temp):
                os.remove(test_path_temp)
            continue
            
    raise FileNotFoundError(f"Could not load or download UCR dataset: {dataset_name}")

if __name__ == "__main__":
    # Test script
    try:
        X_tr, y_tr, X_te, y_te = load_ucr_dataset("GunPoint")
        print(f"Successfully loaded GunPoint dataset:")
        print(f"X_train shape: {X_tr.shape}, y_train shape: {y_tr.shape}")
        print(f"X_test shape: {X_te.shape}, y_test shape: {y_te.shape}")
    except Exception as e:
        print(f"Loader test failed: {e}")
