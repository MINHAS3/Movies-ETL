{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sqlalchemy import create_engine\n",
    "import time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "file_dir = \"C:/Users/Owais/Documents/Bootcamp/\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Owais\\anaconda3\\lib\\site-packages\\IPython\\core\\interactiveshell.py:3254: DtypeWarning: Columns (10) have mixed types.Specify dtype option on import or set low_memory=False.\n",
      "  if (await self.run_code(code, result,  async_=asy)):\n"
     ]
    }
   ],
   "source": [
    "def data_extract(wiki_data, kaggle_metadata, ratings_data): \n",
    "    with open(f'{file_dir}{wiki_data}', mode='r') as file:\n",
    "        wiki_movies_raw = json.load(file)\n",
    "    kaggle_raw = pd.read_csv(f'{file_dir}{kaggle_metadata}')\n",
    "    rating_raw = pd.read_csv(f'{file_dir}{ratings_data}')\n",
    "    return wiki_movies_raw, kaggle_raw, rating_raw\n",
    "\n",
    "wiki_raw, kaggle_raw, ratings_raw = data_extract(\"wikipedia_movies.json\", \"movies_metadata.csv\", \"ratings.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "password = 'zbeast12'\n",
    "location = '127.0.0.1'\n",
    "port = '5432'\n",
    "database = \"movie_data\"\n",
    "user = \"postgres\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'postgres://postgres:zbeast12@127.0.0.1:5432/movie_data'"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sql_connection = f\"postgres://{user}:{password}@{location}:{port}/{database}\"\n",
    "sql_connection "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Engine(postgres://postgres:***@127.0.0.1:5432/movie_data)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "engine = create_engine(sql_connection)\n",
    "engine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_raw.to_sql(name = 'movies', con = engine)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "importing rows 0 to 1000000...Done. 327.66405487060547 total seconds elapsed\n",
      "importing rows 1000000 to 2000000...Done. 595.9120290279388 total seconds elapsed\n",
      "importing rows 2000000 to 3000000...Done. 854.5169360637665 total seconds elapsed\n",
      "importing rows 3000000 to 4000000...Done. 1092.0846724510193 total seconds elapsed\n",
      "importing rows 4000000 to 5000000...Done. 1318.15167760849 total seconds elapsed\n",
      "importing rows 5000000 to 6000000...Done. 1502.103676557541 total seconds elapsed\n",
      "importing rows 6000000 to 7000000...Done. 1708.8036286830902 total seconds elapsed\n",
      "importing rows 7000000 to 8000000...Done. 1889.7130692005157 total seconds elapsed\n",
      "importing rows 8000000 to 9000000...Done. 2072.9014151096344 total seconds elapsed\n",
      "importing rows 9000000 to 10000000...Done. 2260.6582930088043 total seconds elapsed\n",
      "importing rows 10000000 to 11000000...Done. 2461.1498970985413 total seconds elapsed\n",
      "importing rows 11000000 to 12000000...Done. 2646.833548784256 total seconds elapsed\n",
      "importing rows 12000000 to 13000000...Done. 2826.4194598197937 total seconds elapsed\n",
      "importing rows 13000000 to 14000000...Done. 3014.974057674408 total seconds elapsed\n",
      "importing rows 14000000 to 15000000...Done. 3209.0474071502686 total seconds elapsed\n",
      "importing rows 15000000 to 16000000...Done. 6096.893241882324 total seconds elapsed\n",
      "importing rows 16000000 to 17000000...Done. 6374.184784173965 total seconds elapsed\n",
      "importing rows 17000000 to 18000000...Done. 6782.430278301239 total seconds elapsed\n",
      "importing rows 18000000 to 19000000...Done. 7080.012017965317 total seconds elapsed\n",
      "importing rows 19000000 to 20000000...Done. 7383.804991483688 total seconds elapsed\n",
      "importing rows 20000000 to 21000000...Done. 7676.91912817955 total seconds elapsed\n",
      "importing rows 21000000 to 22000000...Done. 7958.640310764313 total seconds elapsed\n",
      "importing rows 22000000 to 23000000...Done. 8236.93929553032 total seconds elapsed\n",
      "importing rows 23000000 to 24000000...Done. 8510.458541154861 total seconds elapsed\n",
      "importing rows 24000000 to 25000000...Done. 8741.47782444954 total seconds elapsed\n",
      "importing rows 25000000 to 26000000...Done. 8948.060055494308 total seconds elapsed\n",
      "importing rows 26000000 to 26024289...Done. 8953.135934352875 total seconds elapsed\n"
     ]
    }
   ],
   "source": [
    "rows_imported = 0\n",
    "# get the start_time from time.time()\n",
    "start_time = time.time()\n",
    "for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):\n",
    "    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')\n",
    "    data.to_sql(name='ratings', con=engine, if_exists='append')\n",
    "    rows_imported += len(data)\n",
    "\n",
    "    # add elapsed time to final print out\n",
    "    print(f'Done. {time.time() - start_time} total seconds elapsed')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "import contextlib\n",
    "from sqlalchemy import MetaData\n",
    "\n",
    "meta = MetaData()\n",
    "\n",
    "with contextlib.closing(engine.connect()) as con:\n",
    "    trans = con.begin()\n",
    "    for table in reversed(meta.sorted_tables):\n",
    "        con.execute(table.delete())\n",
    "    trans.commit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# with open(f'{file_dir}wikipedia_movies.json', mode='r') as file:\n",
    "#     wiki_movies_raw = json.load(file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# First 5 records\n",
    "wiki_raw[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Last 5 records\n",
    "wiki_raw[-5:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Some records in the middle\n",
    "wiki_raw[3600:3605]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata = pd.read_csv(f'{file_dir}movies_metadata.csv')\n",
    "ratings = pd.read_csv(f'{file_dir}ratings.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "# kaggle_metadata = pd.read_csv(\"movies_metadata.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ratings.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df = pd.DataFrame(wiki_movies_raw)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df.head()\n",
    "print(wiki_movies_df.columns)\n",
    "wiki_movies_df.columns.tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# [expression for element in source_list]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# [expression for element in source_list if filter_expression]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# [dog for dog in dogs if dog['weight'] > 30]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# [dog for dog in dogs if dog['is_good'] == True]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# if ('Director' in movie or 'Directed by' in movie) and 'imdb_link' in movie"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies = [movie for movie in wiki_movies_raw\n",
    "               if ('Director' in movie or 'Directed by' in movie)\n",
    "                   and 'imdb_link' in movie]\n",
    "len(wiki_movies)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies = [movie for movie in wiki_movies_raw\n",
    "               if ('Director' in movie or 'Directed by' in movie)\n",
    "                   and 'imdb_link' in movie\n",
    "                   and 'No. of episodes' not in movie]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = 'global value'\n",
    "\n",
    "def foo():\n",
    "    x = 'local value'\n",
    "    print(x)\n",
    "\n",
    "foo()\n",
    "print(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "local value\n",
    "global value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "my_list = [1,2,3]\n",
    "def append_four(x):\n",
    "    x.append(4)\n",
    "append_four(my_list)\n",
    "print(my_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "new_list = list(old_list)\n",
    "new_dict = dict(old_dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lambda arguments: expression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lambda x: x * x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "square = lambda x: x * x\n",
    "square(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_movie(movie): "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_movie(movie):\n",
    "    movie_copy = dict(movie)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_movie(movie):\n",
    "    movie = dict(movie) #create a non-destructive copy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_movie(movie):\n",
    "    movie = dict(movie) #create a non-destructive copy\n",
    "    return movie"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df[wiki_movies_df['Arabic'].notnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df[wiki_movies_df['Arabic'].notnull()]['url']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "6834    https://en.wikipedia.org/wiki/The_Insult_(film)\n",
    "7058     https://en.wikipedia.org/wiki/Capernaum_(film)\n",
    "Name: url, dtype: object"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sorted(wiki_movies_df.columns.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_movie(movie):\n",
    "    movie = dict(movie) #create a non-destructive copy\n",
    "    alt_titles = {}\n",
    "    return movie"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_movie(movie):\n",
    "    movie = dict(movie) #create a non-destructive copy\n",
    "    alt_titles = {}\n",
    "    for key in ['Also known as','Arabic','Cantonese','Chinese','French',\n",
    "                'Hangul','Hebrew','Hepburn','Japanese','Literally',\n",
    "                'Mandarin','McCune–Reischauer','Original title','Polish',\n",
    "                'Revised Romanization','Romanized','Russian',\n",
    "                'Simplified','Traditional','Yiddish']:\n",
    "\n",
    "    return movie"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_movie(movie):\n",
    "    movie = dict(movie) #create a non-destructive copy\n",
    "    alt_titles = {}\n",
    "    for key in ['Also known as','Arabic','Cantonese','Chinese','French',\n",
    "                'Hangul','Hebrew','Hepburn','Japanese','Literally',\n",
    "                'Mandarin','McCune–Reischauer','Original title','Polish',\n",
    "                'Revised Romanization','Romanized','Russian',\n",
    "                'Simplified','Traditional','Yiddish']:\n",
    "        if key in movie:\n",
    "\n",
    "    return movie"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_movie(movie):\n",
    "    movie = dict(movie) #create a non-destructive copy\n",
    "    alt_titles = {}\n",
    "    for key in ['Also known as','Arabic','Cantonese','Chinese','French',\n",
    "                'Hangul','Hebrew','Hepburn','Japanese','Literally',\n",
    "                'Mandarin','McCune–Reischauer','Original title','Polish',\n",
    "                'Revised Romanization','Romanized','Russian',\n",
    "                'Simplified','Traditional','Yiddish']:\n",
    "        if key in movie:\n",
    "            alt_titles[key] = movie[key]\n",
    "            movie.pop(key)\n",
    "\n",
    "    return movie"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_movie(movie):\n",
    "    movie = dict(movie) #create a non-destructive copy\n",
    "    alt_titles = {}\n",
    "    for key in ['Also known as','Arabic','Cantonese','Chinese','French',\n",
    "                'Hangul','Hebrew','Hepburn','Japanese','Literally',\n",
    "                'Mandarin','McCune–Reischauer','Original title','Polish',\n",
    "                'Revised Romanization','Romanized','Russian',\n",
    "                'Simplified','Traditional','Yiddish']:\n",
    "        if key in movie:\n",
    "            alt_titles[key] = movie[key]\n",
    "            movie.pop(key)\n",
    "    if len(alt_titles) > 0:\n",
    "        movie['alt_titles'] = alt_titles\n",
    "\n",
    "    return movie"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "clean_movies = [clean_movie(movie) for movie in wiki_movies]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df = pd.DataFrame(clean_movies)\n",
    "sorted(wiki_movies_df.columns.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def change_column_name(old_name, new_name):\n",
    "        if old_name in movie:\n",
    "            movie[new_name] = movie.pop(old_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "change_column_name('Directed by', 'Director')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_movie(movie):\n",
    "    movie = dict(movie) #create a non-destructive copy\n",
    "    alt_titles = {}\n",
    "    # combine alternate titles into one list\n",
    "    for key in ['Also known as','Arabic','Cantonese','Chinese','French',\n",
    "                'Hangul','Hebrew','Hepburn','Japanese','Literally',\n",
    "                'Mandarin','McCune-Reischauer','Original title','Polish',\n",
    "                'Revised Romanization','Romanized','Russian',\n",
    "                'Simplified','Traditional','Yiddish']:\n",
    "        if key in movie:\n",
    "            alt_titles[key] = movie[key]\n",
    "            movie.pop(key)\n",
    "    if len(alt_titles) > 0:\n",
    "        movie['alt_titles'] = alt_titles\n",
    "\n",
    "    # merge column names\n",
    "    def change_column_name(old_name, new_name):\n",
    "        if old_name in movie:\n",
    "            movie[new_name] = movie.pop(old_name)\n",
    "    change_column_name('Adaptation by', 'Writer(s)')\n",
    "    change_column_name('Country of origin', 'Country')\n",
    "    change_column_name('Directed by', 'Director')\n",
    "    change_column_name('Distributed by', 'Distributor')\n",
    "    change_column_name('Edited by', 'Editor(s)')\n",
    "    change_column_name('Length', 'Running time')\n",
    "    change_column_name('Original release', 'Release date')\n",
    "    change_column_name('Music by', 'Composer(s)')\n",
    "    change_column_name('Produced by', 'Producer(s)')\n",
    "    change_column_name('Producer', 'Producer(s)')\n",
    "    change_column_name('Productioncompanies ', 'Production company(s)')\n",
    "    change_column_name('Productioncompany ', 'Production company(s)')\n",
    "    change_column_name('Released', 'Release Date')\n",
    "    change_column_name('Release Date', 'Release date')\n",
    "    change_column_name('Screen story by', 'Writer(s)')\n",
    "    change_column_name('Screenplay by', 'Writer(s)')\n",
    "    change_column_name('Story by', 'Writer(s)')\n",
    "    change_column_name('Theme music composer', 'Composer(s)')\n",
    "    change_column_name('Written by', 'Writer(s)')\n",
    "\n",
    "    return movie"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "clean_movies = [clean_movie(movie) for movie in wiki_movies]\n",
    "wiki_movies_df = pd.DataFrame(clean_movies)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\\d{7})')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\\d{7})')\n",
    "print(len(wiki_movies_df))\n",
    "wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)\n",
    "print(len(wiki_movies_df))\n",
    "wiki_movies_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "[column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]\n",
    "wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "box_office = wiki_movies_df['Box office'].dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def is_not_a_string(x):\n",
    "    return type(x) != str"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "box_office[box_office.map(is_not_a_string)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lambda arguments: expression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lambda x: type(x) != str"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "box_office[box_office.map(lambda x: type(x) != str)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "some_list = ['One','Two','Three']\n",
    "'Mississippi'.join(some_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "form_one = r'\\$\\d+\\.?\\d*\\s*[mb]illion'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "box_office.str.contains(form_one, flags=re.IGNORECASE).sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "form_two = r'\\$\\d{1,3}(?:,\\d{3})+'\n",
    "box_office.str.contains(form_two, flags=re.IGNORECASE).sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)\n",
    "matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# this will throw an error!\n",
    "box_office[(not matches_form_one) and (not matches_form_two)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "box_office[~matches_form_one & ~matches_form_two]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "form_one = r'\\$\\s*\\d+\\.?\\d*\\s*[mb]illion'\n",
    "form_two = r'\\$\\s*\\d{1,3}(?:,\\d{3})+'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "form_two = r'\\$\\s*\\d{1,3}(?:[,\\.]\\d{3})+'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "form_two = r'\\$\\s*\\d{1,3}(?:[,\\.]\\d{3})+(?!\\s[mb]illion)'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "box_office = box_office.str.replace(r'\\$.*[-—–](?![a-z])', '$', regex=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "form_one = r'\\$\\s*\\d+\\.?\\d*\\s*[mb]illi?on'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "box_office.str.extract(f'({form_one}|{form_two})')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ars(s):\n",
    "    # if s is not a string, return NaN\n",
    "    if type(s) != str:\n",
    "        return np.nan\n",
    "\n",
    "    # if input is of the form $###.# million\n",
    "    if re.match(r'\\$\\s*\\d+\\.?\\d*\\s*milli?on', s, flags=re.IGNORECASE):\n",
    "\n",
    "        # remove dollar sign and \" million\"\n",
    "\n",
    "        # convert to float and multiply by a million\n",
    "\n",
    "        # return value\n",
    "\n",
    "    # if input is of the form $###.# billion\n",
    "    elif re.match(r'\\$\\s*\\d+\\.?\\d*\\s*billi?on', s, flags=re.IGNORECASE):\n",
    "\n",
    "        # remove dollar sign and \" billion\"\n",
    "\n",
    "        # convert to float and multiply by a billion\n",
    "\n",
    "        # return value\n",
    "\n",
    "    # if input is of the form $###,###,###\n",
    "    elif re.match(r'\\$\\s*\\d{1,3}(?:[,\\.]\\d{3})+(?!\\s[mb]illion)', s, flags=re.IGNORECASE):\n",
    "\n",
    "        # remove dollar sign and commas\n",
    "\n",
    "        # convert to float\n",
    "\n",
    "        # return value\n",
    "\n",
    "    # otherwise, return NaN\n",
    "    else:\n",
    "        return np.nan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def parse_dollars(s):\n",
    "    # if s is not a string, return NaN\n",
    "    if type(s) != str:\n",
    "        return np.nan\n",
    "\n",
    "    # if input is of the form $###.# million\n",
    "    if re.match(r'\\$\\s*\\d+\\.?\\d*\\s*milli?on', s, flags=re.IGNORECASE):\n",
    "\n",
    "        # remove dollar sign and \" million\"\n",
    "        s = re.sub('\\$|\\s|[a-zA-Z]','', s)\n",
    "\n",
    "        # convert to float and multiply by a million\n",
    "\n",
    "        # return value\n",
    "\n",
    "    # if input is of the form $###.# billion\n",
    "    elif re.match(r'\\$\\s*\\d+\\.?\\d*\\s*billi?on', s, flags=re.IGNORECASE):\n",
    "\n",
    "        # remove dollar sign and \" billion\"\n",
    "        s = re.sub('\\$|\\s|[a-zA-Z]','', s)\n",
    "\n",
    "        # convert to float and multiply by a billion\n",
    "\n",
    "        # return value\n",
    "\n",
    "    # if input is of the form $###,###,###\n",
    "    elif re.match(r'\\$\\s*\\d{1,3}(?:[,\\.]\\d{3})+(?!\\s[mb]illion)', s, flags=re.IGNORECASE):\n",
    "\n",
    "        # remove dollar sign and commas\n",
    "        s = re.sub('\\$|,','', s)\n",
    "\n",
    "        # convert to float\n",
    "\n",
    "        # return value\n",
    "\n",
    "    # otherwise, return NaN\n",
    "    else:\n",
    "        return np.nan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def parse_dollars(s):\n",
    "    # if s is not a string, return NaN\n",
    "    if type(s) != str:\n",
    "        return np.nan\n",
    "\n",
    "    # if input is of the form $###.# million\n",
    "    if re.match(r'\\$\\s*\\d+\\.?\\d*\\s*milli?on', s, flags=re.IGNORECASE):\n",
    "\n",
    "        # remove dollar sign and \" million\"\n",
    "        s = re.sub('\\$|\\s|[a-zA-Z]','', s)\n",
    "\n",
    "        # convert to float and multiply by a million\n",
    "        value = float(s) * 10**6\n",
    "\n",
    "        # return value\n",
    "        return value\n",
    "\n",
    "    # if input is of the form $###.# billion\n",
    "    elif re.match(r'\\$\\s*\\d+\\.?\\d*\\s*billi?on', s, flags=re.IGNORECASE):\n",
    "\n",
    "        # remove dollar sign and \" billion\"\n",
    "        s = re.sub('\\$|\\s|[a-zA-Z]','', s)\n",
    "\n",
    "        # convert to float and multiply by a billion\n",
    "        value = float(s) * 10**9\n",
    "\n",
    "        # return value\n",
    "        return value\n",
    "\n",
    "    # if input is of the form $###,###,###\n",
    "    elif re.match(r'\\$\\s*\\d{1,3}(?:[,\\.]\\d{3})+(?!\\s[mb]illion)', s, flags=re.IGNORECASE):\n",
    "\n",
    "        # remove dollar sign and commas\n",
    "        s = re.sub('\\$|,','', s)\n",
    "\n",
    "        # convert to float\n",
    "        value = float(s)\n",
    "\n",
    "        # return value\n",
    "        return value\n",
    "\n",
    "    # otherwise, return NaN\n",
    "    else:\n",
    "        return np.nan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df.drop('Box office', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "budget = wiki_movies_df['Budget'].dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "budget = budget.str.replace(r'\\$.*[-—–](?![a-z])', '$', regex=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)\n",
    "matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)\n",
    "budget[~matches_form_one & ~matches_form_two]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df.drop('Budget', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\\s[123]\\d,\\s\\d{4}'\n",
    "date_form_two = r'\\d{4}.[01]\\d.[123]\\d'\n",
    "date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\\s\\d{4}'\n",
    "date_form_four = r'\\d{4}'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "running_time.str.contains(r'^\\d*\\s*minutes$', flags=re.IGNORECASE).sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "running_time[running_time.str.contains(r'^\\d*\\s*minutes$', flags=re.IGNORECASE) != True]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "running_time.str.contains(r'^\\d*\\s*m', flags=re.IGNORECASE).sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "running_time[running_time.str.contains(r'^\\d*\\s*m', flags=re.IGNORECASE) != True]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "running_time_extract = running_time.str.extract(r'(\\d+)\\s*ho?u?r?s?\\s*(\\d*)|(\\d+)\\s*m')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df.drop('Running time', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata['adult'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata['video'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata['video'] == 'True'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)\n",
    "kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')\n",
    "kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ratings.info(null_counts=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.to_datetime(ratings['timestamp'], unit='s')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ratings['rating'].plot(kind='hist')\n",
    "ratings['rating'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df[['title_wiki','title_kaggle']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df[movies_df['release_date_wiki'].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df['Language'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df['original_language'].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df[['Production company(s)','production_companies']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fill_missing_kaggle_data(df, kaggle_column, wiki_column):\n",
    "    df[kaggle_column] = df.apply(\n",
    "        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]\n",
    "        , axis=1)\n",
    "    df.drop(columns=wiki_column, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')\n",
    "fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')\n",
    "fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')\n",
    "movies_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for col in movies_df.columns:\n",
    "    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x\n",
    "    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)\n",
    "    num_values = len(value_counts)\n",
    "    if num_values == 1:\n",
    "        print(col)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df['video'].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',\n",
    "                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',\n",
    "                       'genres','original_language','overview','spoken_languages','Country',\n",
    "                       'production_companies','production_countries','Distributor',\n",
    "                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \\\n",
    "                .rename({'userId':'count'}, axis=1) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \\\n",
    "                .rename({'userId':'count'}, axis=1) \\\n",
    "                .pivot(index='movieId',columns='rating', values='count')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sqlalchemy import create_engine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "engine = create_engine(db_string)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_df.to_sql(name='movies', con=engine)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):\n",
    "    data.to_sql(name='ratings', con=engine, if_exists='append')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create a variable for the number of rows imported\n",
    "for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):\n",
    "\n",
    "    # print out the range of rows that are being imported\n",
    "\n",
    "    data.to_sql(name='ratings', con=engine, if_exists='append')\n",
    "\n",
    "    # increment the number of rows imported by the chunksize\n",
    "\n",
    "    # print that the rows have finished importing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create a variable for the number of rows imported\n",
    "rows_imported = 0\n",
    "for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):\n",
    "\n",
    "    # print out the range of rows that are being imported\n",
    "\n",
    "    data.to_sql(name='ratings', con=engine, if_exists='append')\n",
    "\n",
    "    # increment the number of rows imported by the size of 'data'\n",
    "\n",
    "    # print that the rows have finished importing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create a variable for the number of rows imported\n",
    "rows_imported = 0\n",
    "for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):\n",
    "\n",
    "    # print out the range of rows that are being imported\n",
    "    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')\n",
    "\n",
    "    data.to_sql(name='ratings', con=engine, if_exists='append')\n",
    "\n",
    "    # increment the number of rows imported by the size of 'data'\n",
    "    rows_imported += len(data)\n",
    "\n",
    "    # print that the rows have finished importing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create a variable for the number of rows imported\n",
    "rows_imported = 0\n",
    "for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):\n",
    "\n",
    "    # print out the range of rows that are being imported\n",
    "    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')\n",
    "\n",
    "    data.to_sql(name='ratings', con=engine, if_exists='append')\n",
    "\n",
    "    # increment the number of rows imported by the size of 'data'\n",
    "    rows_imported += len(data)\n",
    "\n",
    "    # print that the rows have finished importing\n",
    "    print('Done.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rows_imported = 0\n",
    "for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):\n",
    "\n",
    "    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')\n",
    "    data.to_sql(name='ratings', con=engine, if_exists='append')\n",
    "    rows_imported += len(data)\n",
    "\n",
    "    print(f'Done.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rows_imported = 0\n",
    "# get the start_time from time.time()\n",
    "for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):\n",
    "\n",
    "    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')\n",
    "    data.to_sql(name='ratings', con=engine, if_exists='append')\n",
    "    rows_imported += len(data)\n",
    "\n",
    "    # add elapsed time to final print out\n",
    "    print(f'Done.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rows_imported = 0\n",
    "# get the start_time from time.time()\n",
    "start_time = time.time()\n",
    "for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):\n",
    "    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')\n",
    "    data.to_sql(name='ratings', con=engine, if_exists='append')\n",
    "    rows_imported += len(data)\n",
    "\n",
    "    # add elapsed time to final print out\n",
    "    print(f'Done.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rows_imported = 0\n",
    "# get the start_time from time.time()\n",
    "start_time = time.time()\n",
    "for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):\n",
    "    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')\n",
    "    data.to_sql(name='ratings', con=engine, if_exists='append')\n",
    "    rows_imported += len(data)\n",
    "\n",
    "    # add elapsed time to final print out\n",
    "    print(f'Done. {time.time() - start_time} total seconds elapsed')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}