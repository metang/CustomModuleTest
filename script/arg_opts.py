# Movie Lens User Data arguments parser
import argparse

def movie_data_opts():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_size",
                        default='100k',
                        type=str,
                        required=True,
                        help="The size of movie data to retrieve. Has to be one of the 4 values: 100k, 1m, 10m, 20m.")
    parser.add_argument("--user_data_path",
                        default="output",
                        type=str,
                        required=False,
                        help="Output of retrieved user data.")
    parser.add_argument("--include_title",
                        default="Title",
                        type=str,
                        required=False,
                        help="whether to add title column in the retrieved data")
    parser.add_argument("--include_genre",
                        default="Genre",
                        type=str,
                        required=False,
                        help="whether to add genre column in the retrieved data")
    parser.add_argument("--include_year",
                        default="Year",
                        type=str,
                        required=False,
                        help="whether to add year column in the retrieved data")


    return parser

