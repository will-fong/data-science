import luigi

class HelloLuigi(luigi.Task):

    def output(self):
        return luigi.LocalTarget('hello-luigi.txt')

    def run(self):
        with self.output().open("w") as outfile:
            outfile.write("Hello Luigi!")

# To execute the task you created, run the following command:
# python -m luigi --module hello-world HelloLuigi --local-scheduler

# Alternatively add PYTHONPATH='.' to the front of your Luigi command:
# PYTHONPATH='.' luigi --module hello-world HelloLuigi --local-scheduler

# With the --module hello-world HelloLuigi flag, Luigi will know which Python module and Luigi task to execute.
# Running tasks using the local-scheduler flag is only recommended for development work.
