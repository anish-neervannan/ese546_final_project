import numpy as np
import pandas as pd

table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]

# Uncomment to save
# df.to_csv("data/S&P500.csv", columns=['Symbol', 'GICS Sector'])

res = {}
for index, row in df.iterrows():
    if row['GICS Sector'] not in res:
        res[row['GICS Sector']] = [row['Symbol']]
    else:
        res[row['GICS Sector']].append(row['Symbol'])


res = {}
for index, row in df.iterrows():
    if row['GICS Sector'] not in res:
        res[row['GICS Sector']] = [row['Symbol']]
    else:
        res[row['GICS Sector']].append(row['Symbol'])


num_total_stocks = 0
num_days = 750
raw_sector_data = {}
for sector in res:
    sector_data = []
    num_sector_stocks = 0
    for ticker in res[sector]:
        try:
            data = pd.read_csv(f"stocks/{ticker.lower()}.us.txt")
            stock_data = data.iloc[-1 * num_days:,]['Open'].to_numpy()
            
            if (stock_data.shape[0] == num_days):
                sector_data.append(stock_data)
                num_sector_stocks += 1
                num_total_stocks += 1
        except:
            pass
    sector_data = np.concatenate(sector_data, axis=0).reshape((num_sector_stocks, num_days))
    np.save(f"data/train_and_val_{sector}", sector_data[:,0:int(num_days * 0.8)])
    np.save(f"data/test_{sector}", sector_data[:,int(num_days * 0.8):])    

print(f"Total Stocks: {num_total_stocks}")

sector = "Information Technology"
raw_train_X = np.load(f"data/train_and_val_{sector}.npy")
print(raw_train_X.shape)