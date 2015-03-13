import numpy as np
import pandas as pd
from subprocess import call

# Averages over Kaggle submission files
def combine_mean(filenames):
	dfs = []
	print "Reading files..."
	for f in filenames:
		dfs.append(pd.DataFrame.from_csv(f))

	print "Averaging..."

	# Start averaging
	df_concat = pd.concat(dfs)
	df_mean = df_concat.groupby(level=0).mean()

	print "Writing to file..."

	df_mean.to_csv('out_averaged.csv')

	print "Gzipping..."

	call("gzip -c out_averaged.csv > out_averaged.csv.gz", shell=True)

	print "Done! Upload file out_averaged.csv.gz to Kaggle."

if __name__ == "__main__":
	combine_mean(['MODEL_CONCURRENT.csv', 'MODEL_MORE_FILTERS.csv', 'MODEL_CONCURRENT_FLIPS_100.csv'])