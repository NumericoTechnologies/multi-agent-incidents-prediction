"""Create adjacency matrix by start & end points of segments"""
import os
import sys
import pandas as pd
from collections import OrderedDict

this_dir = os.path.abspath(os.path.dirname(__file__))
projects_dir = os.path.dirname(this_dir)
if projects_dir not in sys.path:
    sys.path.append(projects_dir)

from common_utils.general_utils import csv_to_df

input = '/Users/recep/Documents/NumericoTech/multi_agent_incidents_prediction/data/segmentation/intersecting_points_numericoid.csv'
output = '/Users/recep/Documents/NumericoTech/multi_agent_incidents_prediction/data/adjacency_matrix.csv'

df = csv_to_df(input, sep=',')
numerico_ids = df.numerico_i.unique()

adjacent_segments_dict = dict()

for numerico_id in numerico_ids:
    segment_points = df.loc[(df.numerico_i == numerico_id)]
    segments = list()
    for index, row in segment_points.iterrows():
        adjacent_segments = df.loc[(row['xcoord'] == df.xcoord) & (
                row['ycoord'] == df.ycoord), :]
        adjacent_segments = adjacent_segments.loc[
            (adjacent_segments.numerico_i != numerico_id)]
        adjacent_segments = adjacent_segments.drop_duplicates('numerico_i')
        for i in adjacent_segments['numerico_i'].tolist():
            segments.append(i)
        segments = list(OrderedDict.fromkeys(segments))
    adjacent_segments_dict[numerico_id] = segments

g = {k: [v.strip() for v in vs] for k, vs in adjacent_segments_dict.items()}
edges = [(a, b) for a, bs in g.items() for b in bs]
df = pd.DataFrame(edges)
adj_matrix = pd.crosstab(df[0], df[1])
adj_matrix.to_csv(output)

