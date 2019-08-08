# This is the custom module code for loading moving rating data
import os
import json
import sys
import pyarrow.parquet as pq

sys.path.append("../")
from reco_utils.dataset.movielens import load_pandas_df
from script.arg_opts import movie_data_opts

class MovieLensData:
    def __init__(self, meta: dict = {}):
        self.data_size = str(meta.get('Data Size', '100k'))
        self.include_title = 'Title' if meta.get('Include Title', 'False') == 'True' else None
        self.include_genre = 'Genre' if meta.get('Include Genre', 'False') == 'True' else None
        self.include_year = 'Year' if meta.get('Include Year', 'False') == 'True' else None

    def run(self, meta: dict = None):
        df_userdata = load_pandas_df(self.data_size, ('UserId', 'ItemId', 'Rating', 'Timestamp'),
            title_col=self.include_title,
            genres_col=self.include_genre,
            year_col=self.include_year
        )
        return df_userdata

def main():
    parser = movie_data_opts()
    args, _ = parser.parse_known_args()

    meta = {'Data Size': str(args.data_size),
            'Include Title': str(args.include_title),
            'Include Genre': str(args.include_genre),
            'Include Year': str(args.include_year),
            'User Data Path': str(args.user_data_path)}
    
    print(meta)

    movie_data = MovieLensData(meta=meta)
    df_userdata = movie_data.run()

    if not os.path.exists(args.user_data_path):
        os.makedirs(args.user_data_path)
    df_userdata.to_parquet(fname=os.path.join(args.user_data_path, args.data_size + ".parquet"), engine='pyarrow')

    # Dump data_type.json as a work around until SMT deploys
    dct = {
        "Id": "Dataset",
        "Name": "Dataset .NET file",
        "ShortName": "Dataset",
        "Description": "A serialized DataTable supporting partial reads and writes",
        "IsDirectory": False,
        "Owner": "Microsoft Corporation",
        "FileExtension": "dataset.parquet",
        "ContentType": "application/octet-stream",
        "AllowUpload": False,
        "AllowPromotion": True,
        "AllowModelPromotion": False,
        "AuxiliaryFileExtension": None,
        "AuxiliaryContentType": None
    }
    with open(os.path.join(args.user_data_path, 'data_type.json'), 'w') as f:
        json.dump(dct, f)


if __name__ == "__main__":
    main()
