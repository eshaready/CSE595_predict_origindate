import pandas as pd
import csv
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
# FINAL FORMAT FOR DATA:
# year, book or movie, words

# Read movie data delimited svs into pandas dfs
movietitles = pd.read_csv("movie_titles_metadata.txt", delimiter = " \+\+\+\$\+\+\+ ", engine='python', names=["movie_id", "movie_title", "release_year", "rating", "votes", "genres"])
movielines = pd.read_csv("movie_lines.txt", delimiter = " \+\+\+\$\+\+\+ ", encoding= 'unicode_escape', engine='python', names=["line_id", "character_id", "movie_id", "speaker", "line"])
movieinfo = pd.merge(movietitles, movielines, on="movie_id", how="inner")
movieinfo = movieinfo[["release_year", "line"]]

# Read the harvard data jsons into pandas dfs
books = pd.read_json("NO COPYRIGHT - UNITED STATES-0000.json", lines=True)
files = [
    "NO COPYRIGHT - UNITED STATES-0001.jsonl",
    "IN COPYRIGHT-0000.jsonl",
    "IN COPYRIGHT-0001.jsonl",
]

for f in files:
    df = pd.read_json(f, lines=True)
    books = pd.concat([books, df], ignore_index=True)
books = books[books["gxml_language"] == "eng"]
books = books[["gxml_date_2", "gxml_date_1", "text_by_page"]]

# print(movieinfo.columns, "\n", movieinfo.head(5))
# print(books.head(5))

# Format tables into the correct formats
# Add "type": "movie" column to movieinfo
movieinfo["type"] = "movie"
# Add "type": "book" column to books
books["type"] = "book"

# Compare gxml_date_2 and gxml_date_1; if they are both integers, take the average. If one is,
# take that as the date. if neither is, remove the column
def choose_year(x):
    if x["gxml_date_2"].isdigit() and x["gxml_date_1"].isdigit():
        return int((int(x["gxml_date_2"]) + int(x["gxml_date_1"])) / 2)
    elif x["gxml_date_2"].isdigit():
        return int(x["gxml_date_2"])
    elif x["gxml_date_1"].isdigit():
        return int(x["gxml_date_1"])
    else:
        return None
books["year"] = books.apply(lambda x: choose_year(x), axis=1) 
books = books[["year", "text_by_page", "type"]]

# Format lines better: want to break the list into separate rows
books = books.explode("text_by_page")

# Drop any none values
books["text_by_page"].replace('', np.nan, inplace=True)
books = books.dropna()
movieinfo["line"].replace('', np.nan, inplace=True)
movieinfo = movieinfo.dropna()

# Merge into one table 
movieinfo = movieinfo.rename(columns={"release_year": "year"})
books = books.rename(columns={"text_by_page": "line"})
combined = pd.concat([books, movieinfo], axis=0)
combined["year"] = pd.to_numeric(combined["year"], errors="coerce")
combined = combined.dropna()

# This is just to find distribution information
combined["decade"] = (combined["year"] // 10) * 10
combined["century"] = (combined["year"] // 100) * 100
print("Describe:", combined.describe())
print("Decade distribution:", combined["decade"].value_counts())
print("Century distribution:", combined["century"].value_counts())

# # ---- SAVE TO FILE
# # split into train val test
# combined["line"] = combined["line"].replace('\n', '\\n', regex=True)
# main, test = train_test_split(combined, test_size=0.1, random_state=12)
# train, val = train_test_split(main, test_size=0.1, random_state=12)

# # save to csv 
# train.to_csv("train.csv", encoding='utf-8', index=False, quoting=csv.QUOTE_ALL)
# test.to_csv("test.csv", encoding='utf-8', index=False, quoting=csv.QUOTE_ALL)
# val.to_csv("val.csv", encoding='utf-8', index=False, quoting=csv.QUOTE_ALL)