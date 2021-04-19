import requests
import luigi
from bs4 import BeautifulSoup

from collections import Counter
import pickle

class GetTopBooks(luigi.Task):
    """
    Get list of the most popular books from Project Gutenberg
    """

    def output(self):
        return luigi.LocalTarget("data/books_list.txt")

    def run(self):
        # Download the HTML contents of the Project Gutenberg top books page
        resp = requests.get("http://www.gutenberg.org/browse/scores/top")

        # Parse the page contents
        soup = BeautifulSoup(resp.content, "html.parser")

        # Find the header for top 100
        pageHeader = soup.find_all("h2", string="Top 100 EBooks yesterday")[0]
        listTop = pageHeader.find_next_sibling("ol")

        # Loop over the structure to get all of the links. For this page, locate all links <a> that are within a list item <li>. For each of those links, if they link to a page that points at a link containing /ebooks/, assume it is a book and write that link to your output() file.
        with self.output().open("w") as f:
            for result in listTop.select("li>a"):
                if "/ebooks/" in result["href"]:
                    f.write("http://www.gutenberg.org{link}.txt.utf-8\n"
                        .format(
                            link=result["href"]
                        )
                    )

class DownloadBooks(luigi.Task):
    """
    Download a specified list of books
    """
    # Specify a line in your list of URLs to fetch
    FileID = luigi.IntParameter()

    REPLACE_LIST = """.,"';_[]:*-"""

    # Define prerequisite task
    def requires(self):
        return GetTopBooks()

    # Create a dynamic file name from the File ID parameter
    def output(self):
        return luigi.LocalTarget("data/downloads/{}.txt".format(self.FileID))

    # Retrieve the URL from the line,  download the contents, iterate through the characters list to replace with a space, standardize capitalization by converting to lower case, and write to output
    def run(self):
        with self.input().open("r") as i:
            URL = i.read().splitlines()[self.FileID]

            with self.output().open("w") as outfile:
                book_downloads = requests.get(URL)
                book_text = book_downloads.text

                for char in self.REPLACE_LIST:
                    book_text = book_text.replace(char, " ")

                book_text = book_text.lower()
                outfile.write(book_text)

class CountWords(luigi.Task):
    """
    Count the frequency of the most common words from a file
    """

    FileID = luigi.IntParameter()

    def requires(self):
        return DownloadBooks(FileID=self.FileID)

    def output(self):
        return luigi.LocalTarget(
            "data/counts/count_{}.pickle".format(self.FileID),
            format=luigi.format.Nop
        )

    # Open the downloaded file, count the words, and save into a pickle output
    def run(self):
        with self.input().open("r") as i:
            word_count = Counter(i.read().split())

            with self.output().open("w") as outfile:
                pickle.dump(word_count, outfile)

# Parameters for books and words to analyze
class GlobalParams(luigi.Config):
    NumberBooks = luigi.IntParameter(default=10)
    NumberTopWords = luigi.IntParameter(default=500)

class TopWords(luigi.Task):
    """
    Aggregate the count results from the different files
    """

    def requires(self):
        requiredInputs = []
        for i in range(GlobalParams().NumberBooks):
            requiredInputs.append(CountWords(FileID=i))
        return requiredInputs

    def output(self):
        return luigi.LocalTarget("data/summary.txt")

    # Store the count, "unpickle" the count, continuously sum the count, and write the most common words
    def run(self):
        total_count = Counter()
        for input in self.input():
            with input.open("rb") as infile:
                nextCounter = pickle.load(infile)
                total_count += nextCounter

        with self.output().open("w") as f:
            for item in total_count.most_common(GlobalParams().NumberTopWords):
                f.write("{0: <15}{1}\n".format(*item))

# To execute the task you created, run the following command:
# python -m luigi --module word-frequency GetTopBooks --local-scheduler

# To view the list, run the following command:
# cat data/books_list.txt

# Run after starting luigid
# python -m luigi --module word-frequency DownloadBooks --FileID 2

# Count the words
# python -m luigi --module word-frequency CountWords --FileID 2

# Analyze the word counts
# python -m luigi --module word-frequency TopWords --GlobalParams-NumberBooks 15 --GlobalParams-NumberTopWords 750

# To view the list, run the following command:
# cat data/summary.txt