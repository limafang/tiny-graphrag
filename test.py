from utils import save_triplets_to_txt
from groq import Chatbot
from graph import get_triplets, Neo4jHandler, split_text, get_entity
from tqdm import tqdm
import os

uri = "bolt://localhost:7687"
user = "neo4j"
password = "Fangshiyi0"

handler = Neo4jHandler(uri, user, password)

folder_path = "data"
entity_list = []
segment_list = []
# read folder
for root, dirs, files in os.walk(folder_path):
    for file in files:
        print(file)
        file_path = os.path.join(root, file)
        segments = split_text(file_path)
        for segment in tqdm(segments):
            entity = get_entity(segment)
            entity_list.append(entity)
            segment_list.append(segment)
        break
print("=" * 30)
print(entity_list)
print(segment_list)


for i, entity in enumerate(entity_list):
    if entity:
        # print(entity)
        triplets = get_triplets(text=segment, entity=entity)
        if triplets:
            print(triplets)
            for triplet in triplets:
                # print(triplet)
                handler.create_triplet(triplet[0], triplet[1], triplet[2])

handler.close()

# read file
# file_path = "data\IMPLEMENTATION.md"
# segments = split_text(file_path)

# res = []
# for segment in tqdm(segments):
#     # pre process TODO
#     entity = get_entity(segment)
#     if entity:
#         print(entity)
#         triplets = get_triplets(text=segment, entity=entity)
#         if triplets:
#             print(triplets)
#             for i in triplets:
#                 print(i)
#                 handler.create_triplet(i[0], i[1], i[2])
# handler.close()
