#%%
import numpy as np
from bioservices import KEGG
from Bio.KEGG.REST import kegg_get
from Bio.KEGG.KGML import KGML_parser
import xml.etree.ElementTree as ET
import networkx as nx
import pickle
import re
# %%
# 11 primarily signaling pathways in cancer
primarily_signaling_pathways = ["Central carbon metabolism in cancer",
                                    "Choline metabolism in cancer",
                                    "Sphingolipid signaling pathway",
                                    "mTOR signaling pathway",
                                    "Adrenergic signaling in cardiomyocytes",
                                    "VEGF signaling pathway",
                                    "Apelin signaling pathway",
                                    "TNF signaling pathway",
                                    "Retrograde endocannabinoid signaling",
                                    "GnRH signaling pathway",
                                    "Oxytocin signaling pathway",
                                ]
# Initialize KEGG client
k = KEGG()
k.organism = 'hsa'
for pathway in primarily_signaling_pathways:
    search_results = k.find("pathway", pathway)
    print(search_results)

#%%
hsa_pathway_list = ['hsa05230','hsa05231','hsa04071',
                    'hsa04150','hsa04261','hsa04370',
                    'hsa04371','hsa04668','hsa04723',
                    'hsa04912','hsa04921']
graphics_dict_list = []
relations_list = []
G_list = []
for pathway in hsa_pathway_list:
    pathway_kgml = k.get(pathway, "kgml")
    root = ET.fromstring(pathway_kgml) 
    # Initialize gene list and dictionary to map entry IDs to genes
    gene_dict = {}
    relations = []
    graphics_dict = {}
    # Extract genes and map entry IDs to gene names
    for entry in root.findall("entry"):
        if entry.get("type") == "gene":
            entry_id = entry.get("id")
            gene_names = entry.get("name").split()  # Splits 'hsa:6513 hsa:6514' into ['hsa:6513', 'hsa:6514']
            gene_dict[entry_id] = gene_names
            graphics = entry.find("graphics")
            if graphics is not None:
                graphics_name = graphics.get("name").split(',')[0]  # Retrieve the 'name' attribute from the graphics element
                graphics_name = re.sub(r'\.\.\.$', '', graphics_name)
                graphics_dict[entry_id] = graphics_name
            else:
                graphics_dict[entry_id] = None
    
    for relation in root.findall("relation"):
        entry1 = relation.get("entry1")
        entry2 = relation.get("entry2")
        if entry1 not in gene_dict or entry2 not in gene_dict:
            continue
        subtypes = [subtype.get("name") for subtype in relation.findall("subtype")]
        if "activation" in subtypes or "inhibition" in subtypes:  # Simplified condition for weight
            weight = 1 if "activation" in subtypes else -1
        else:
            weight = 0
        relations.append((entry1, entry2, weight))
    relations_list.append(relations)
    graphics_dict_list.append(graphics_dict)
    G = nx.DiGraph()

    # Add edges to the graph
    for entry1, entry2, weight in relations:
        G.add_edge(entry1, entry2, weight=weight)

    # If your graph should be undirected, convert it
    G = G.to_undirected()

    G_list.append(G)

    # nodes = list(G.nodes())
    # for i in nodes:
    #     print(graphics_dict[i])
with open('G_list.pkl', 'wb') as f:
    pickle.dump(G_list, f)
with open('relations_list.pkl', 'wb') as f:
    pickle.dump(relations_list, f)
with open('graphics_dict_list.pkl', 'wb') as f:
    pickle.dump(graphics_dict_list, f)

# %%
