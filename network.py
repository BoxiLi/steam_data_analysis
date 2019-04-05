import csv
import json
import networkx as nx
import matplotlib.pyplot as plt

id_file_name = "steam_data3000"

steam_id_set = set()
with open(id_file_name + ".csv", 'r') as f:
    reader = csv.reader(f, delimiter = '\n')

    for row in reader:
        try:
            steam_id_set.add(int(row[0]))
        except IndexError:
            pass


json_data = open(id_file_name + ".json").read() 
steam_info_data = json.loads(json_data)


steam_net = nx.Graph()
steam_net.add_nodes_from(steam_id_set)
for user in steam_info_data: # careful, user is a str
    user_int = int(user)
    edge_list = []
    fr_list = steam_info_data[user]["friends"]
    for fr in fr_list: # fr is an int
        if fr in steam_id_set:
            edge_list.append((user_int, fr))
    steam_net.add_edges_from(edge_list)

# steam_net.remove_nodes_from(list(nx.isolates(steam_net)))

options = {
     'node_color': 'black',
     'node_size': 5,
     'width': 1,
     'edge_color' : 'blue',
     'alpha' : 0.4
}
plt.subplot(111)
nx.draw(steam_net, with_labels=False, **options)
plt.show()