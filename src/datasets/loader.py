import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd

def load_local_ucr_txt(file_path):
    try:
        data = np.loadtxt(file_path)
        y = data[:, 0]
        X = data[:, 1:]
        return X, y
    except Exception as e:
        df = pd.read_csv(file_path, header=None, sep=None, engine='python')
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        return X, y

def load_ts_file(file_path):
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
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    train_path_txt = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.txt")
    test_path_txt = os.path.join(dataset_dir, f"{dataset_name}_TEST.txt")
    train_path_tsv = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.tsv")
    test_path_tsv = os.path.join(dataset_dir, f"{dataset_name}_TEST.tsv")
    train_path_ts = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.ts")
    test_path_ts = os.path.join(dataset_dir, f"{dataset_name}_TEST.ts")
    
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
        pass

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
        zip_path_temp = os.path.join(dataset_dir, f"{dataset_name}.zip")
        if os.path.exists(zip_path_temp):
            os.remove(zip_path_temp)

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
        
    mirrors = [
        (f"https://raw.githubusercontent.com/sktime/sktime/main/sktime/datasets/data/{dataset_name}/{dataset_name}_TRAIN.ts", train_path_ts, ".ts"),
        (f"https://raw.githubusercontent.com/hfawaz/cd-diagram/master/{dataset_name}/{dataset_name}_TRAIN.tsv", train_path_tsv, ".tsv"),
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
            if os.path.exists(local_path):
                os.remove(local_path)
            test_path_temp = local_path.replace("TRAIN", "TEST")
            if os.path.exists(test_path_temp):
                os.remove(test_path_temp)
            continue
            
    raise FileNotFoundError(f"Could not load or download UCR dataset: {dataset_name}")
