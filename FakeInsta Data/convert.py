import pandas as pd

def main():
    with open('fakeAccountData.json', encoding='utf-8-sig') as f:
        df = pd.read_json(f)
    df.to_csv('fakeAccount.csv', encoding='utf-8', index=False)

    with open('realAccountData.json', encoding='utf-8-sig') as f:
        df2 = pd.read_json(f)
    df2.to_csv('readAccountData.csv', encoding='utf-8', index=False)
    

if __name__ == "__main__":
    main()