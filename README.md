# fantasy-football

Attempt at modeling fantasy football player projections based on past performances using an LSTM. 

1. data_collection notebook shows how to scrape from profootball reference
2. clean_data cleans modeling data found from gridironai.com. The cleaned data sets per position are available in fantasy_data. 
3. predictions(last 16 games predictions), actuals, and 2021 projections(trained with entire dataset) are in their respective folders 




Note: Code does take awhile to run, I highly recommend using Google Colab with GPU or TPU (link for colab and GPU use https://colab.research.google.com/notebooks/gpu.ipynb). Also recommend putting in points in code to automatically write to csv after every 50 players or so to ensure no progress is lost. 
