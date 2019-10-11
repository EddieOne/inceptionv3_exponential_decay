import tensorflow as tf, sys, os

# Unpersists graph from file
with tf.gfile.FastGFile("graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

    for i in tf.get_default_graph().get_operations():
        print i.name

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("labels.txt")]
#label_lines.reverse()
print(label_lines)

def predict(image_data, filename):
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f) %s' % (human_string, score, filename))
            break

# Read the images, send data tp prediction model
for (dirpath, dirs, files) in os.walk('test/'):
    for filename in files:
        image_data = tf.gfile.FastGFile('test/' + filename, 'rb').read()
        predict(image_data, filename)
