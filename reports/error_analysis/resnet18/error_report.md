# ResNet18 错误分析

## 总体情况

- 测试图片数：1291
- 正确数量：1266
- 错误数量：25
- 测试准确率：0.9806
- 错误样本平均置信度：0.4824

## 主要混淆组合

| 真实类别 | 预测类别 | 数量 |
| --- | --- | ---: |
| Tomato___Late_blight | Tomato___Early_blight | 13 |
| Tomato___Late_blight | Tomato___healthy | 2 |
| Tomato___Leaf_Mold | Tomato___Early_blight | 2 |
| Apple___healthy | Tomato___healthy | 1 |
| Apple___healthy | Apple___Apple_scab | 1 |
| Apple___Apple_scab | Apple___healthy | 1 |
| Tomato___Early_blight | Apple___healthy | 1 |
| Tomato___Late_blight | Apple___healthy | 1 |
| Tomato___Early_blight | Tomato___healthy | 1 |
| Tomato___Early_blight | Tomato___Late_blight | 1 |

## 表现最差的类别

| 类别 | 正确数 | 总数 | 准确率 |
| --- | ---: | ---: | ---: |
| Tomato___Late_blight | 268 | 284 | 0.9437 |
| Tomato___Leaf_Mold | 141 | 144 | 0.9792 |
| Tomato___Early_blight | 147 | 150 | 0.9800 |
| Apple___Apple_scab | 94 | 95 | 0.9895 |
| Apple___healthy | 244 | 246 | 0.9919 |