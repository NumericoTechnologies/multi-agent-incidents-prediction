import argparse
import os
from common_utils.general_utils import csv_to_df, df_to_csv


numerico_ids = ['284312011#284312008', '284312010', '282312031', '282312022',
                '282312021', '281312036#281311068#280311003',
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


def filter_incidents(incidents_csv, out_dir, numerico_ids):
    df = csv_to_df(incidents_csv, sep='|', parse_dates=['starttime', 'stoptime'])
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument('--function', required=True,
                        choices=['filter_incidents'],
                        help='Select function for preprocessing')
    parser.add_argument('--out_dir',
                        default='data/',
                        type=str, help='Output files directory')
    parser.add_argument('--incidents_csv', type=str,
                        help='Path to incidents_csv')
    args = parser.parse_args()
    kwargs_ = dict(vars(args))
    if args.function == 'filter_incidents':
        filter_incidents(args.incidents_csv, args.out_dir, numerico_ids)

