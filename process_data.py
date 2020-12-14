import pandas as pd

table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]

# Uncomment to save
# df.to_csv("S&P500.csv", columns=['Symbol', 'GICS Sector'])

res = {}
for index, row in df.iterrows():
    if row['GICS Sector'] not in res:
        res[row['GICS Sector']] = [row['Symbol']]
    else:
        res[row['GICS Sector']].append(row['Symbol'])


count = 0
for sector in res:
    for ticker in res[sector]:
        try:
            data = pd.read_csv(f"Stocks/{ticker.lower()}.us.txt")
            count += 1
            print(data.iloc[-1]["Date"])
        except:
            print(f"{ticker}: Ticker Not Found")

print(f"Total Stocks: {count}")