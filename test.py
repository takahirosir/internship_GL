import spacy
import networkx as nx
import matplotlib.pyplot as plt

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google, few people outside of the company took him seriously.")

doc = nlp(text)

# Create a directed graph
G = nx.DiGraph()

# Add nodes to the graph
for entity in doc.ents:
    G.add_node(entity.text, label=entity.label_)

# Add edges to the graph
for token in doc:
    for child in token.children:
        if token.ent_type_ and child.ent_type_:
            G.add_edge(token.text, child.text)

# Draw the graph
pos = nx.spring_layout(G)
labels = {node: data['label'] for node, data in G.nodes(data=True)}
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_labels(G, pos, labels)
plt.show()