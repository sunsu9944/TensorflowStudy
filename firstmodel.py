import tensorflow as tf

month6 = [60,80,50,80,100,30]
month9 = [80,100,90,90,100,50]
result = [70,90,70,85,100,40]

w1 = tf.Variable(tf.random.normal([1]))
w2 = tf.Variable(tf.random.normal([1]))
w3 = tf.Variable(tf.random.normal([1]))
w4 = tf.Variable(tf.random.normal([1]))

w5 = tf.Variable(tf.random.normal([1]))
w6 = tf.Variable(tf.random.normal([1]))

#model을 만든다

def loss(a,b,c,d,e,f):
    
    hidden1 = a * month6 + b * month9
    hidden2 = c * month6 + d * month9

    model = hidden1 * e + hidden2 * f


    return tf.keras.losses.mse(result,model)


opt = tf.keras.optimizers.Adam(learning_rate=0.5)

for i in range(3000):
    opt.minimize(lambda : loss(w1,w2,w3,w4,w5,w6) , var_list=[w1,w2,w3,w4,w5,w6])
    print(w1.numpy(),w2.numpy(),w3.numpy(),w4.numpy(),w5.numpy(),w6.numpy())


real6 = 25
real9 = 90


realresult = (w1 * real6 + w2 * real9)*w5 + (w3 * real6 + w4 * real9)*w6


print(realresult)



