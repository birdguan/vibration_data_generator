# 振动信号生成
## 数据
![振动信号数据](imgs/data/img_0.png)  
振动信号(256×256×3)  
## 模型
- DCGAN (standard GAN | Relativistic GAN)
- WGAP-GP
## 运行

### 项目结构
|- frozen_model  
&nbsp;&nbsp;&nbsp;&nbsp;|-- 模型存放处  
|- gray_imgs         
&nbsp;&nbsp;&nbsp;&nbsp;|-- data  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- 灰度训练数据存放处  
|- imgs     
&nbsp;&nbsp;&nbsp;&nbsp;|-- data  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- 彩色训练数据存放处  
|- result  
&nbsp;&nbsp;&nbsp;&nbsp;|-- 生成结果  
|- src  
&nbsp;&nbsp;&nbsp;&nbsp;|-- dcgan.py DCGAN模型及训练测试代码  
&nbsp;&nbsp;&nbsp;&nbsp;|-- parameters.py 训练参数  
&nbsp;&nbsp;&nbsp;&nbsp;|-- wgan-gp.py WGAN-GP模型及训练测试代码  
&nbsp;&nbsp;&nbsp;&nbsp;|-- 其余代码，未使用   
|- training_dcgan  
&nbsp;&nbsp;&nbsp;&nbsp;|-- DCGAN训练中间变量
## 使用dcgan
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

## 结论
实验证明，DCGAN效果是要好于WGAN-GP的。   
另256×256的数据确实比较大，DCGAN也显得捉襟见肘，可使用PGGAN。