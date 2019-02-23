# tensorflow_tips

## 混合精度

1. 当你创建变量时，请使用tf.float16

```python
    dtype = tf.float16
    data = tf.placeholder(dtype, shape=(nbatch, nin))
    weights = tf.get_variable('weights', (nin, nout), dtype)
    biases = tf.get_variable('biases', nout, dtype,
                            initializer=tf.zeros_initializer())
    logits = tf.matmul(data, weights) + biases
```

2. 确保需要训练的变量是float32精度，然后在模型中使用它们时转换为float16

```python
    tf.cast(tf.get_variable(..., dtype=tf.float32), tf.float16)
```

3. 确保loss函数的精度为float32

```python
    tf.losses.softmax_cross_entropy(target, tf.cast(logits, tf.float32))
```

4. 应用loss-scaling，在计算梯度的时候乘以比例因子，一般为128，然后将得到的梯度除以相同的比例

```python
    loss, params = ...
    scale = 128
    grads = [grad / scale for grad in tf.gradients(loss * scale, params)]
```
