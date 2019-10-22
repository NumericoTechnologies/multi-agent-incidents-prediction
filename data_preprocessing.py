import argparse
import os
import datetime
import sys
import csv

this_dir = os.path.abspath(os.path.dirname(__file__))
projects_dir = os.path.dirname(this_dir)
if projects_dir not in sys.path: sys.path.append(projects_dir)

from common_utils.general_utils import csv_to_df, df_to_csv

numerico_ids_list = ['284312011#284312008', '284312010', '282312031',
                     '282312022', '282312021', '281312036#281311068#280311003',
                     '281312032#282312026#281312039#281312034#281312035#280312010',
                     '281312031#282312025#282312033#284312013', '280312016',
                     '280312013#279312012', '280312011',
                     '280311004#279311039#279311040',
                     '279312009#278312011#280312014#279312013#279312011#279312003',
                     '279312008#279312007#278311021', '279312004',
                     '279311042#279311043',
                     '279311036#278311037#279311041#278312002#278311038',
                     '279311034', '279311024',
                     '279311009#278311013#278311023#278311034',
                     '278313088#278313089#278313045', '278313080',
                     '278313074#278313075#277315030', '278313046',
                     '278312014#278312007#278312003#278313086#278313091#278313078#278313048#278312012',
                     '278312013#278311035#277311059#278311041#278311030#278311022#278311040',
                     '278311025#278311027#278311028#277311081#277311079',
                     '278309025#278309026#279311035#279310048',
                     '278309023#278309022#278309024',
                     '278309011#278309010#279311029',
                     '278308002#278308001#278309019#278309021',
                     '277315032#277316021#277317004', '277315031',
                     '276318051#275317063#275317060#276318023#277317005#276318043',
                     '276318046#276318045']

top_features_list = ['original_speed_min', 'diffx_speed_periodStart_max',
                     'diffx_speed_specificLane_mean', 'diffx_speed_order_mean',
                     'original_flow_mean', 'diffx_speed_order_max',
                     'diffx_speed_specificLane_max']


def filter_incidents(incidents_csv, out_dir, numerico_ids):
    df = csv_to_df(incidents_csv, sep='|',
                   parse_dates=['starttime', 'stoptime'])
    len_df = len(df)
    df['Numerico_ID'] = ''
    for index, row in df.iterrows():
        for numerico_id in numerico_ids:
            if str(row['WVK_ID']) in numerico_id:
                df.set_value(index, 'Numerico_ID', numerico_id)
    df = df.drop(df.index[df['Numerico_ID'] == ''])
    len_gps, old_len = len(df), len_df
    print(f'{old_len - len_gps} lines dropped without matching ids')
    df_to_csv(df, os.path.join(out_dir, 'incidents_numerico_id.csv'), sep='|')


def create_training_data(file_dir, incidents_csv, out_dir, top_features, past,
                         future):
    out_folder = os.path.join(out_dir, 'train_data')
    os.makedirs(out_folder, exist_ok=True)
    segment_features = sorted([f for f in os.listdir(file_dir)])
    inc_df = csv_to_df(incidents_csv, sep='|',
                       parse_dates=['starttime', 'stoptime'])
    for features in segment_features:
        segment_ID = features[:-13]
        ft_df = csv_to_df(os.path.join(file_dir, features), sep='|',
                          parse_dates=['periodStart'])
        # Select only top feature columns
        ft_df = ft_df[['periodStart'] + top_features]
        # impute NaNs with 0
        ft_df = ft_df.fillna(0.)
        # normalize values
        for feature in top_features:
            ft_df[feature] = (ft_df[feature] - ft_df[feature].min()) / (
                    ft_df[feature].max() - ft_df[feature].min())
        # get all incidents with segment_ID
        inc_df_seg = inc_df.drop(
            inc_df.index[inc_df['Numerico_ID'] != segment_ID])
        with open(os.path.join(out_folder, segment_ID + '.csv'), 'w',
                  newline='') as output:
            writer = csv.writer(output, delimiter='|')
            for i in range(past - 1, len(ft_df)):
                if i % 10000 == 0:
                    print(f'{datetime.datetime.now()}: {i} entries processed.')
                timestamp_min = ft_df.loc[i][0]
                timestamp_max = timestamp_min + datetime.timedelta(
                    minutes=future)
                # get all incidents between timestamp min & max
                inc_df_seg_cur = inc_df_seg.loc[
                             (timestamp_min < inc_df_seg.starttime) & (
                                     inc_df_seg.starttime <= timestamp_max), :]
                # create list with n past timestamps
                values = sum([list(ft_df.loc[i - j][1:].values) for j in
                              range(past - 1, -1, -1)], [])
                values.insert(0, timestamp_min)
                if not inc_df_seg_cur.empty:
                    values.append(1.)
                else:
                    values.append(0.)
                writer.writerow(values)


def csv_to_hdf5(file_dir, out_dir):
    out_folder = os.path.join(out_dir, 'train_data_hdf5')
    os.makedirs(out_folder, exist_ok=True)
    segment_features = sorted([f for f in os.listdir(file_dir)])
    print(f'Saving files in {out_folder}')
    for features in segment_features:
        df = csv_to_df(os.path.join(file_dir, features), sep='|', header=None)
        df.to_hdf(os.path.join(out_folder, features[:-4] + '.h5'), 'data',
                  mode='w', format='table')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument('--function', required=True,
                        choices=['filter_incidents', 'create_training_data',
                                 'csv_to_hdf5'],
                        help='Select function for preprocessing')
    parser.add_argument('--out_dir', default='data/', type=str,
                        help='Output files directory')
    parser.add_argument('--incidents_csv', type=str,
                        help='Path to incidents_csv')
    parser.add_argument('--segments_feature_dir', type=str,
                        help='Directory containing features of segments')
    parser.add_argument('--past', type=int, default=5,
                        help='data point consists of n past timestamps')
    parser.add_argument('--future', type=int, default=20,
                        help='data point takes incidents happening up to n '
                             'future minutes into account')
    args = parser.parse_args()
    kwargs_ = dict(vars(args))
    if args.function == 'filter_incidents':
        filter_incidents(args.incidents_csv, args.out_dir, numerico_ids_list)
    if args.function == 'create_training_data':
        create_training_data(args.segments_feature_dir, args.incidents_csv,
                             args.out_dir, top_features_list, args.past,
                             args.future)
    if args.function == 'csv_to_hdf5':
        csv_to_hdf5(args.segments_feature_dir, args.out_dir)