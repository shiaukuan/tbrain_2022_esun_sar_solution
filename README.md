# tbrain_2022_esun_sar_1st_solution
2022 玉山人工智慧公開挑戰賽冬季賽。你說可疑不可疑？－疑似洗錢交易預測 
[比賽連結](https://tbrain.trendmicro.com.tw/Competitions/Details/24)

## 檔案用途:
- Preprocess/: 存放前處理的code
- Model/: 存放模型相關code
- requirements.txt: 需要的套件


## 執行流程:
#### 安裝所需套件
```sh
$ conda create --name <env> --file requirements.txt
```
#### 執行資料前處理生成特徵
```sh
Preprocess$ python feature_extraction.py
```
#### 訓練模型、預測
```sh
Model$ python train.py
Model$ python inference.py 
```

![image](https://user-images.githubusercontent.com/5851454/210141489-2453a512-278f-4947-9823-10e2c7a0f357.png)
