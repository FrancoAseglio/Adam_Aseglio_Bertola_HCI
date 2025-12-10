import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

## Comment or uncomment the appropriate data path
# INPUT_DIR_PATH = "C:\\Users\\Lorenzo\\OneDrive\\Desktop\\datasets\\cleaned\\"
INPUT_DIR_PATH = "~/datasets/cleaned/"

## files to read format: adapt to pd.read_format
FILE_EXTENSION = ".csv"

# character_anime_works_df = pd.read_csv(INPUT_DIR_PATH + "character_anime_works_clean" + FILE_EXTENSION)
# character_nicknames_df = pd.read_csv(INPUT_DIR_PATH + "character_nicknames_clean" + FILE_EXTENSION)
# characters_df = pd.read_csv(INPUT_DIR_PATH + "characters_clean" + FILE_EXTENSION)
# details_df = pd.read_csv(INPUT_DIR_PATH + "details_clean" + FILE_EXTENSION)
# favs_df = pd.read_csv(INPUT_DIR_PATH + "favs_clean" + FILE_EXTENSION)
# person_alternate_names_df = pd.read_csv(INPUT_DIR_PATH + "person_alternate_names_clean" + FILE_EXTENSION)
# person_anime_works_df = pd.read_csv(INPUT_DIR_PATH + "person_anime_works_clean" + FILE_EXTENSION)
# person_details_df = pd.read_csv(INPUT_DIR_PATH + "person_details_clean" + FILE_EXTENSION)
# person_voice_works_df = pd.read_csv(INPUT_DIR_PATH + "person_voice_works_clean" + FILE_EXTENSION)
# profiles_df = pd.read_csv(INPUT_DIR_PATH + "profiles_clean" + FILE_EXTENSION)
# ratings_df = pd.read_csv(INPUT_DIR_PATH + "ratings_clean" + FILE_EXTENSION)
# recommendations_df = pd.read_csv(INPUT_DIR_PATH + "recommendations_clean" + FILE_EXTENSION)
# stats_df = pd.read_csv(INPUT_DIR_PATH + "stats_clean" + FILE_EXTENSION)
