import pandas as pd
import numpy as np
import nltk
from nltk.corpus import words
from gensim.models import KeyedVectors
import gensim.downloader as api
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

from stellargraph import StellarGraph
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt

df = pd.read_csv('urlset/urlset.csv', error_bad_lines=False, encoding='latin1')
urls = df['domain'].to_numpy()
labels = df['label'].to_numpy()

d = set(words.words())
word_vectors = api.load("glove-wiki-gigaword-100")  

common_words = set()
common_words_df = pd.read_csv('google-10000-english.txt', sep=' ')
for word in list(common_words_df.to_numpy().flatten()[:5000]):
    common_words.add(word)

print('data loaded')

DATASET_PCT = 1

#Preprocess & randomize data, this is so we avoid the first n urls all being labelled 1
urls_short = []
labels_short = []
for u in range(len(urls)):
    n = np.random.rand()
    if n < DATASET_PCT:
        urls_short.append(urls[u])
        labels_short.append(labels[u])
    
label_dict= {}
for i in range(len(urls_short)):
    label_dict[urls_short[i]] = labels_short[i]
    
# map urls to 
node_index = {}
index_node = {}
label_index = {}
i = 0
for n in urls_short:
    node_index[n] = i
    index_node[i] = n
    if (n in label_dict) and (not np.isnan(label_dict[n])):
        label_index[i] = int(label_dict[n])
    else:
        label_index[i] = 0
    i+=1

def find_words(subs):
    
    i = 0
    j = 4
    w = set()
    for s in subs:
        if len(s) < 4:
            continue
        for i in range(len(s) - 1):
            for j in range(3, len(s) - i):
                #print(s[i:i+j+1], s[i:i+j+1] in d)
                if s[i:i+j+1] in d and s[i:i+j+1] in common_words:
                    w.add(s[i:i+j+1])
    return w

#dict of url to similar words
sim_words_dict = {}

print('Finding Similar Words')
t=1
for i in range(len(urls_short)):
    if i > (len(urls_short))/(10/t):
        print(f'{t*10}%', end= ' ')
        t+=1
    url = str(urls_short[i])
    
    #finds the related words
    sub_domains = url.split('/')[1:]
    w = find_words(sub_domains)
    
    words = set(w)
    #will go through and find the similar words for each word in w
    for j in w:
        j = str(j)
        try:
            similar = set((np.asarray(word_vectors.most_similar([j], topn=3, restrict_vocab=None)).T)[0])
        except:
            #set it up as an empty set if there are no similar words
            similar = set()
        
        words |= similar
        
    sim_words_dict[node_index[url]] = words

print('Creating Graph')

# Define our graph with connections between urls that have similar words
G = nx.Graph()
t = 1

for i in range(len(urls_short)-1):
    for j in range(i+1, len(urls_short)):
        if i*j > (len(urls_short)**2)/(10/t):
            print(f'{t*10}%', end= ' ')
            t+=1
        
        url1 = node_index[urls_short[i]]
        url2 = node_index[urls_short[j]]
        
        if url1 == url2:
            continue
        
        if url1 not in G:
            G.add_node(url1)
        if url2 not in G:
            G.add_node(url2)

        if len(sim_words_dict[url1].intersection(sim_words_dict[url2])) > 5 and (not G.has_edge(url1, url2)):
            G.add_edge(url1, url2, weight = len(sim_words_dict[url1].intersection(sim_words_dict[url2]))/100)

nx.write_adjlist(G, 'urls_only.adjlist')

print('Creating Word Vectors')

word_dict = set()
for n in sim_words_dict:
    for w in sim_words_dict[n]:
        word_dict.add(w)

sim_word_vecs = {}
for n in G.nodes:
    l = []
    for w in word_dict:
        # create a vector for each node with 1s for each similar word
        if w in sim_words_dict[n]:
            l.append(1)
        else:
            l.append(0)
    sim_word_vecs[n] = l

fs = []
nodes = []

for n in G.nodes:
    nodes.append(n)
    fs.append(sim_word_vecs[n])

features = pd.DataFrame(fs, index=nodes)

SG = StellarGraph.from_networkx(G, node_features = features)
print(SG.info())
features.to_csv('features.csv')

url_set = pd.Series(label_index)
url_set.to_csv('url_set.csv')
train_nodes, test_nodes = model_selection.train_test_split(url_set, train_size=(len(url_set)*8)//10, test_size=None, stratify=url_set)

hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256

x_train = []
y_train = []
for ind in list(train_nodes.keys()):
    x_train.append(sim_word_vecs[ind])
    y_train.append(label_index[ind])
    
    
x_test = []
y_test = []
for ind in list(test_nodes.keys()):
    x_test.append(sim_word_vecs[ind])
    y_test.append(label_index[ind])
    
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
print(x_train.shape)
print(y_train.shape)
print(sum(y_train))
print(x_test.shape)
print(y_test.shape)
print(sum(y_test))

def run_experiment(model, x_train, y_train):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )

    return history

def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()

def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)

def create_baseline_model(hidden_units, num_classes, dropout_rate=0.2):
    inputs = layers.Input(shape=(num_features,), name="input_features")
    x = create_ffn(hidden_units, dropout_rate, name=f"ffn_block1")(inputs)
    for block_idx in range(4):
        # Create an FFN block.
        x1 = create_ffn(hidden_units, dropout_rate, name=f"ffn_block{block_idx + 2}")(x)
        # Add skip connection.
        x = layers.Add(name=f"skip_connection{block_idx + 2}")([x, x1])
    # Compute logits.
    logits = layers.Dense(num_classes, name="logits")(x)
    # Create the model.
    return keras.Model(inputs=inputs, outputs=logits, name="baseline")

num_classes=2
num_features = x_train.shape[1]
baseline_model = create_baseline_model(hidden_units, num_classes, dropout_rate)
baseline_model.summary()

history = run_experiment(baseline_model, x_train, y_train)

x,test_accuracy = baseline_model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Baseline Test accuracy: {test_accuracy * 100}%")

edges = []
weights = []
for e in G.edges():
    edges.append(list(e))
    weights.append(G.get_edge_data(e[0], e[1])['weight'])
    
edges = np.asarray(edges).T
edge_weights = np.asarray(weights)
node_features = features.to_numpy().astype('float64')
print(edges.shape)
print(edge_weights.shape)
print(node_features.shape)
graph_info = (node_features, edges, edge_weights)

NUM_NODES = node_features.shape[0]
class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        if self.combination_type == "gated":
            self.update_fn = layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        num_nodes = NUM_NODES
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_repesentations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(node_indices, neighbour_messages)
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)

class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super(GNNNodeClassifier, self).__init__(*args, **kwargs)

        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        # Create a process layer.
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )
        # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2",
        )
        # Create a postprocess layer.
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        # Create a compute logits layer.
        self.compute_logits = layers.Dense(units=num_classes, name="logits")

    def call(self, input_node_indices):
        # Preprocess the node_features to produce node representations.
        x = self.preprocess(self.node_features)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x1 + x
        # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x2 + x
        # Postprocess node embedding.
        x = self.postprocess(x)
        # Fetch node embeddings for the input node_indices.
        node_embeddings = tf.gather(x, input_node_indices)
        # Compute logits
        return self.compute_logits(node_embeddings)

gnn_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=2,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
    name="gnn_model",
)

print("GNN output shape:", gnn_model([1, 10, 100]))

gnn_model.summary()

x_train = np.asarray(train_nodes.keys())
history = run_experiment(gnn_model, x_train, y_train)

x_test = np.asarray(test_nodes.keys())
x,test_accuracy = gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"GCN Test Accuracy: {test_accuracy * 100}%")