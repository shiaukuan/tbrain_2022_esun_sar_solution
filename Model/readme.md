# 模型訓練
本資料夾內含三個程式:
- train.py: 訓練模型
- inference.py: 使用訓練好的模型進行預測，並產生繳交的csv檔
- sffs.py: 修改mlxtend的SequentialFeatureSelector方法，挑選出的特徵讓模型預測更穩定

```sh
Model$ python train.py
Model$ python inference.py 
```
## sffs.py 修改部分

[原本程式](https://github.com/rasbt/mlxtend/blob/1f46260b18cf569cec5b9c487e4448c5f2c37630/mlxtend/feature_selection/sequential_feature_selector.py)

挑選變數
```python
for new_subset, cv_scores in work:
    all_avg_scores.append(np.nanmean(cv_scores))
    all_cv_scores.append(cv_scores)
    all_subsets.append(new_subset)

if len(all_avg_scores) > 0:
    best = np.argmax(all_avg_scores)
    out = (all_subsets[best], all_avg_scores[best], all_cv_scores[best])
```

挑選變數由平均修改為最差10筆平均(使用30次交叉驗證)
```python
for new_subset, cv_scores in work:
    
    all_avg_scores.append(np.nanmean(sorted(cv_scores)[:10]) )

    all_cv_scores.append(cv_scores)
    all_subsets.append(new_subset)

if len(all_avg_scores) > 0:
    best = np.argmax(all_avg_scores)
    out = (all_subsets[best], all_avg_scores[best], all_cv_scores[best])
```

一開始
```python
self.subsets_[k] = {
    "feature_idx": k_idx,
    "cv_scores": k_score,
    "avg_score": np.nanmean(k_score),
}
```
改成
```python
self.subsets_[k] = {
    "feature_idx": k_idx,
    "cv_scores": k_score,
    "avg_score": np.nanmean(sorted(k_score)[:10]),
}
```
