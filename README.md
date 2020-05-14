# 振动信号生成
## 数据
![振动信号数据](imgs/data/img_0.png)  
振动信号  
## 模型
- DCGAN (standard GAN | Relativistic GAN)
- WGAP-GP
## 运行
### 使用dcgan
- 训练  
```python
设置 istrain = True  
python src/dcgan.py
```
- 测试
```python
设置 istrain = False  
python src/dcgan.py
```

### 使用wgan-gp
- 训练  
```python
设置 istrain = True  
python src/wgan-gp.py
```
- 测试
```python
设置 istrain = False  
python src/wgan-gp.py
```

## 结果
![](training_dcgan/img_generated_epoch_110.png)