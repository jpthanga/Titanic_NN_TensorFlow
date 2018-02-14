import tensorflow as tf
import numpy as np
import pandas as pd

id = np.load("id.npy")
labs = np.load("labels.npy")
dat = np.load("data.npy")

t_dat = np.load("t_data.npy")
t_id = np.load("t_id.npy")

features = tf.convert_to_tensor(dat,dtype=tf.float32)
label = tf.convert_to_tensor(labs,dtype=tf.float32)

t_features = tf.convert_to_tensor(t_dat,dtype=tf.float32)

x = tf.shape(label)

l1w = tf.Variable(tf.truncated_normal([8,50]))
l1b = tf.Variable(tf.zeros([50]))

l2w = tf.Variable(tf.truncated_normal([50,2]))
l2b = tf.Variable(tf.zeros([2]))

def letc(dat):
    inp = tf.matmul(dat,l1w)
    hid = tf.nn.relu(inp+l1b)
    out = tf.matmul(hid,l2w)+l2b
    return out

scores = letc(features)

# loss = tf.losses.mean_squared_error(labels=label, predictions=scores)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=label))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

pred = tf.round(tf.nn.softmax(letc(t_features)))

# #
init = tf.global_variables_initializer()
# #
sess = tf.Session()
sess.run(init)
for i in range(10000):
  _, loss_value = sess.run((train, loss))
  # print(loss_value)
  if(loss_value<0.2):
      break

prediction = np.array(sess.run(pred))
prediction = prediction[:,0].reshape(-1,1)

out = pd.DataFrame(data=t_id,columns=['PassengerId'])
out['Survived'] = prediction.astype(int)

out.to_csv("final.csv",columns=['PassengerId','Survived'],index=False)

print(out)



