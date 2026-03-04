# MovieLens-1M Dataset

## 来源
- **官网**: https://grouplens.org/datasets/movielens/1m/
- **发布机构**: GroupLens Research, University of Minnesota
- **许可**: 仅供研究使用

## 数据规模

| 指标 | 数值 |
|------|------|
| 评分总数 | 1,000,209 |
| 用户数 | 6,040 |
| 电影数 | 3,883 |
| 稀疏度 | 95.74% |
| 用户平均交互 | 165.6 条 |
| 评分范围 | 1-5 (整数) |

## 文件说明

### ratings.dat
- 格式: `UserID::MovieID::Rating::Timestamp`
- 每位用户至少有 20 条评分记录

### users.dat
- 格式: `UserID::Gender::Age::Occupation::Zip-code`
- **Gender**: M=Male, F=Female
- **Age**: 1=Under 18, 18=18-24, 25=25-34, 35=35-44, 45=45-49, 50=50-55, 56=56+
- **Occupation**: 0-20 共 21 种职业

### movies.dat
- 格式: `MovieID::Title::Genres`
- Genres 为 `|` 分隔的多标签

## 项目适配说明

### 冷启动模拟策略
- 每个用户视为一个 meta-learning task
- 随机抽取 K 条交互作为 support set（模拟冷启动场景）
- 剩余交互作为 query set（评估泛化性能）
- K ∈ {5, 10, 20} 模拟不同冷启动程度

### 稀疏ID特征（验证梯度消失/步长解耦）
- `user_id`: 6,040 个, 分布均匀
- `movie_id`: 3,883 个, 长尾分布
- `age`: 7 组
- `gender`: 2 类
- `occupation`: 21 种 (其中 farmer=17人, 低频ID验证)

### 低频ID分布（验证梯度补偿机制）
| Occupation | 用户数 | 标记 |
|-----------|--------|------|
| 8 (farmer) | 17 | ⚠️ 极低频 |
| 18 (retired) | 70 | ⚠️ 低频 |
| 19 (sales) | 72 | ⚠️ 低频 |
| 9 (homemaker) | 92 | ⚠️ 低频 |

## 评分分布

| Rating | Count | 占比 |
|--------|-------|------|
| 1 | 56,174 | 5.6% |
| 2 | 107,557 | 10.8% |
| 3 | 261,197 | 26.1% |
| 4 | 348,971 | 34.9% |
| 5 | 226,310 | 22.6% |

## 引用
```bibtex
@article{harper2015movielens,
  title={The MovieLens Datasets: History and Context},
  author={Harper, F. Maxwell and Konstan, Joseph A.},
  journal={ACM Transactions on Interactive Intelligent Systems},
  volume={5},
  number={4},
  pages={1--19},
  year={2015}
}
```
