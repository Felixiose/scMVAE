.PHONY = conda download_data run



conda:
	# Make sure to install miniconda first.
	conda update conda
	conda env create -f environment.yml
download_data:
	pip install --no-deps geoopt==0.1.0
	python -m data.download_data
run: 
	python -m Semester_Project.examples.run


