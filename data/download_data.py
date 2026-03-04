"""
数据集下载与预处理
支持 MovieLens-1M 数据集
"""
import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path


MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens_1m(data_dir: str = "./data/ml-1m"):
    """下载 MovieLens-1M 数据集"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    ratings_file = data_path / "ratings.dat"
    if ratings_file.exists():
        print(f"[INFO] 数据集已存在: {ratings_file}")
        return str(data_path)
    
    zip_path = data_path.parent / "ml-1m.zip"
    
    if not zip_path.exists():
        print(f"[INFO] 正在下载 MovieLens-1M ...")
        response = requests.get(MOVIELENS_1M_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    print(f"\r[DOWNLOAD] {pct:.1f}%", end="", flush=True)
        print("\n[INFO] 下载完成")
    
    # 解压
    print("[INFO] 正在解压...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path.parent)
    
    print(f"[INFO] 数据集已准备好: {data_path}")
    return str(data_path)


def load_movielens_1m(data_dir: str = "./data/ml-1m"):
    """加载 MovieLens-1M 数据集"""
    ratings_file = os.path.join(data_dir, "ratings.dat")
    
    if not os.path.exists(ratings_file):
        download_movielens_1m(data_dir)
    
    # 读取评分数据
    ratings = pd.read_csv(
        ratings_file,
        sep="::",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1"
    )
    
    # 读取用户数据
    users_file = os.path.join(data_dir, "users.dat")
    users = pd.read_csv(
        users_file,
        sep="::",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        engine="python",
        encoding="latin-1"
    )
    
    # 读取电影数据
    movies_file = os.path.join(data_dir, "movies.dat")
    movies = pd.read_csv(
        movies_file,
        sep="::",
        header=None,
        names=["item_id", "title", "genres"],
        engine="python",
        encoding="latin-1"
    )
    
    print(f"[INFO] 加载完成: {len(ratings)} 条评分, {ratings['user_id'].nunique()} 用户, {ratings['item_id'].nunique()} 电影")
    
    return ratings, users, movies


def preprocess_data(ratings: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame):
    """
    数据预处理：
    1. 重编码 user_id, item_id 为连续整数
    2. 将评分二值化（>=4 为正样本）
    3. 构建用户特征、物品特征
    """
    # 重编码ID
    user_id_map = {uid: idx for idx, uid in enumerate(sorted(ratings["user_id"].unique()))}
    item_id_map = {iid: idx for idx, iid in enumerate(sorted(ratings["item_id"].unique()))}
    
    ratings = ratings.copy()
    ratings["user_idx"] = ratings["user_id"].map(user_id_map)
    ratings["item_idx"] = ratings["item_id"].map(item_id_map)
    
    # 二值化评分
    ratings["label"] = (ratings["rating"] >= 4).astype(np.float32)
    
    # 用户特征编码
    gender_map = {"F": 0, "M": 1}
    age_map = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    
    users = users.copy()
    users["gender_idx"] = users["gender"].map(gender_map)
    users["age_idx"] = users["age"].map(age_map)
    users["user_idx"] = users["user_id"].map(user_id_map)
    
    # 物品特征编码 - 提取genres
    all_genres = set()
    for g in movies["genres"]:
        all_genres.update(g.split("|"))
    genre_list = sorted(all_genres)
    genre_map = {g: idx for idx, g in enumerate(genre_list)}
    
    movies = movies.copy()
    movies["item_idx"] = movies["item_id"].map(item_id_map)
    movies["genre_indices"] = movies["genres"].apply(
        lambda x: [genre_map[g] for g in x.split("|")]
    )
    
    # 统计用户交互频率
    user_freq = ratings.groupby("user_idx").size().reset_index(name="freq")
    
    # 统计物品交互频率
    item_freq = ratings.groupby("item_idx").size().reset_index(name="freq")
    
    info = {
        "num_users": len(user_id_map),
        "num_items": len(item_id_map),
        "num_ratings": len(ratings),
        "num_genres": len(genre_list),
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
        "genre_list": genre_list,
        "user_freq": user_freq,
        "item_freq": item_freq,
    }
    
    print(f"[INFO] 预处理完成: {info['num_users']} users, {info['num_items']} items, {info['num_ratings']} interactions")
    
    return ratings, users, movies, info


if __name__ == "__main__":
    download_movielens_1m("./data/ml-1m")
